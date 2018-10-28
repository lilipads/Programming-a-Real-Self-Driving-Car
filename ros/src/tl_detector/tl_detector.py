#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import numpy as np
import math


# only call the detector when the waypoint index of the car is closer
# than this distance to the next traffic light
TL_DETECTION_DISTANCE_THRESHOLD = 100
# only change state after three consecutive frames have detected the
# same traffic light state
STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        # Combining queue_size and buff_size to minimize lag, see https://answers.ros.org/question/220502/image-subscriber-lag-despite-queue-1/
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1, buff_size=10000000)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.light_score_confidence_threshold = 0.4  # when score is above this threshold, we are confident about the prediction
        self.light_score_unknown_threashold = 0.2  # when score is below this threshold, the light is classified as unknown
        #for matching tensorflow class (R:1, G:2, Y:3) with TrafficLight msg (R:0, Y:1, G:2)
        self.light_class_dict = {1:0, 2:2, 3:1}
        #TrafficLight msg (R:0, Y:1, G:2, Un:4) to names
        self.light_name_dict = {0:'Red_traffic_light',
                                1:'Yellow_traffic_light',
                                2:'Green_traffic_light',
                                4:'Unkown'}
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoint_tree = KDTree([[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                            for waypoint in waypoints.waypoints])

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state, score = self.process_traffic_lights()
        rospy.loginfo('Current detection state: {}, Light waypoint index: {}'.format(
                                            self.light_name_dict[state],
                                            light_wp))
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to eiher occur `STATE_COUNT_THRESHOLD` number
        of times or has a high predicted score till we start using it.
        Otherwise the previous stable state is used.
        '''
        if (self.state != state) and (score < self.light_score_confidence_threshold):
            self.state_count = 1
            self.state = state
        elif (self.state_count >= STATE_COUNT_THRESHOLD) or (
            score >= self.light_score_confidence_threshold):
            self.last_state = self.state
            light_wp = light_wp if state in (
                TrafficLight.RED, TrafficLight.YELLOW) else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

        if self.last_wp == -1:
            rospy.loginfo('Execution: continue driving')
        else:
            rospy.loginfo('Execution: STOPPING')

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Calling the classifier to get the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
            score: score of predicted traffic light state. Higher prediction score means higher confidence.
        """
        if not self.has_image:
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        # expand_dims to shape [1, None, None, 3].
        image_expanded = np.expand_dims(cv_image, axis=0)

        t_start = rospy.get_time()
        boxes, scores, classes, num  = self.light_classifier.get_classification(image_expanded)
        classification_time = rospy.get_time() - t_start

        rospy.loginfo('Detected Class: {}, Score: {:.4f},  Inference time: {:.4f}s'.format(
                                                self.light_name_dict[self.light_class_dict[classes[0][0]]],
                                                scores[0][0],
                                                classification_time))
        if num > 0 and scores[0][0] > self.light_score_unknown_threashold:
            return self.light_class_dict[classes[0][0]], scores[0][0]

        return TrafficLight.UNKNOWN, 0

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
            float: predicted score of the traffic light state. Higher score means higher prediction confidence.
        """
        closest_light = None
        light_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            rospy.loginfo("Car waypoint idx: " + str(car_wp_idx))
            # find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                #get stopline waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    light_wp_idx = temp_wp_idx

        do_detection = self.is_within_detection_distance(self.waypoints.waypoints, car_wp_idx, light_wp_idx)

        if closest_light and do_detection:
            state, score = self.get_light_state(closest_light)
            return light_wp_idx, state, score
        else:
            if closest_light:
                rospy.loginfo("Did not run detector. Next light is too far away.")
            return -1, TrafficLight.UNKNOWN, 0

    def is_within_detection_distance(self, waypoints, wp1, wp2):
        """
        return true if the distance between the two waypoints is
        close enough to activate the traffic light classifier.

        """
        if wp1 is None or wp2 is None:
            return True
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        # jumping every 3 wps because we just need a coarse estimate
        for i in range(wp1, wp2+1, 3):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
            if dist > TL_DETECTION_DISTANCE_THRESHOLD:
                return False
        return True


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
