#!/usr/bin/env python

import rospy
import math
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped
from styx_msgs.msg import Lane

class Visualize(object):
    def __init__(self):
        rospy.init_node('visualize')

        self.final_waypoints = None
        self.current_pose = None

        rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_cb)
        rospy.Subscriber('/final_waypoints', Lane, self.final_waypoints_cb)

        self.marker_publisher = rospy.Publisher('/marker_array', Marker, queue_size=1)
        self.marker_object = Marker()
        self.marker_object.header.frame_id = "/world"
        self.marker_object.header.stamp = rospy.get_rostime()
        self.marker_object.ns = "visualize"
        self.marker_object.id = 0
        self.marker_object.type = Marker.SPHERE_LIST
        self.marker_object.action = Marker.ADD
        self.marker_object.scale.x = 0.5
        self.marker_object.scale.y = 0.5
        self.marker_object.scale.z = 0.5
        self.marker_object.color.r = 0.0
        self.marker_object.color.g = 1.0
        self.marker_object.color.b = 0.0
        self.marker_object.color.a = 1.0

        # Duration(0): infinite lifetime
        self.marker_object.lifetime = rospy.Duration(0)

        self.loop()

    def loop(self):
        """
        publish marker of final waypoints from car's perspective
        """
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.current_pose and self.final_waypoints:
                self.marker_object.points = []
                for i, wp in enumerate(self.final_waypoints.waypoints):
                    roll, pitch, yaw = euler_from_quaternion([self.current_pose.pose.orientation.x,
                                                              self.current_pose.pose.orientation.y,
                                                              self.current_pose.pose.orientation.z,
                                                              self.current_pose.pose.orientation.w])
                    temp_point = Point()
                    #get relative waypoint position(x, y) from current position
                    p_x = wp.pose.pose.position.x - self.current_pose.pose.position.x
                    p_y = wp.pose.pose.position.y - self.current_pose.pose.position.y
                    #perspective transform (map coord -> car coord)
                    #(+) direction of x axis->yaw=0, Clock-wise->yaw (+), Counter Clock-wise->yaw (-)
                    cos_yaw = math.cos(-yaw)
                    sin_yaw = math.sin(-yaw)
                    temp_point.x = p_x * cos_yaw - p_y * sin_yaw
                    temp_point.y = p_x * sin_yaw + p_y * cos_yaw
                    temp_point.z = 0.0
                    self.marker_object.points.append(temp_point)

                self.marker_publisher.publish(self.marker_object)
            rate.sleep()

    def current_pose_cb(self, msg):
        self.current_pose = msg

    def final_waypoints_cb(self, waypoints):
        self.final_waypoints = waypoints

if __name__ == '__main__':
    try:
        Visualize()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start visualize node.')
