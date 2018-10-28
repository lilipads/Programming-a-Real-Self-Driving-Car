import rospy
import tensorflow as tf
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #path to frozen pb file based on model parameter from styx/site launch files
        self.traffic_light_model = rospy.get_param('traffic_light_model')
        rospy.loginfo('Initializing traffic light classifier with model %s', self.traffic_light_model)
        self.model_path = 'model/' + self.traffic_light_model  + '/frozen_inference_graph.pb'
        self.graph = tf.get_default_graph()
        self.graph_def = tf.GraphDef()
        with tf.gfile.GFile(self.model_path, 'rb') as fid:
            serialized_graph = fid.read()
            self.graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(self.graph_def, name='')
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """
        Determines traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            boxes, scores, classes, num
        """
        boxes, scores, classes, num = self.sess.run([self.detection_boxes,
                                                     self.detection_scores,
                                                     self.detection_classes,
                                                     self.num_detections],
                                                     feed_dict={self.image_tensor: image})
        return boxes, scores, classes, num
