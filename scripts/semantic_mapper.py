#!/usr/bin/env python3

import rospy
import message_filters
import numpy as np
import tf
import argparse
from semantic_predictor_segformer import SemanticPredictor
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from threading import Lock


object_name_to_id = {
    'chair': 0,
    'bed': 1,
    'plant': 2,
    'toilet': 3,
    'tv_monitor': 4,
    'sofa': 5
}


class SemanticMapper:
    def __init__(self):
        rospy.init_node('semantic_mapper')
        image_topic_1 = rospy.get_param('~image_topic_1', 'image_1')
        image_topic_2 = rospy.get_param('~image_topic_2', '')
        semantic_mask_topic_1 = rospy.get_param(
            '~semantic_mask_topic_1', 'semantic_mask_1')
        semantic_mask_topic_2 = rospy.get_param(
            '~semantic_mask_topic_2', 'semantic_mask_2')
        vision_range = rospy.get_param('~vision_range', 5.0)
        self.target_classes = rospy.get_param(
            '~target_classes', 'sofa').split(',')
        self.goal_decay = rospy.get_param('~goal_decay', 0.8)
        self.goal_threshold = rospy.get_param('~goal_threshold', 2.0)
        self.erosion_size = rospy.get_param('~erosion', 4)
        self.vision_range = int(vision_range * 100)
        self.agent_height = 1.0
        self.agent_view_angle = 0.0
        self.bridge = CvBridge()
        self.semantic_predictor = SemanticPredictor()
        self.depths = []
        self.obstacle_map = None
        self.semantic_map = None
        self.map_corner_position = (0, 0)
        self.map_resolution = 5
        self.z_bins = [25, 150]
        self.f = 910.4
        self.cx = 648.4
        self.cy = 354.0
        self.left_side = None

        self.last_processed_stamp = rospy.Time()
        self.min_time_passed = rospy.Duration(0)
        if image_topic_2 == '':
            self.image_subscriber = rospy.Subscriber(
                image_topic_1, CompressedImage, self.image_callback, queue_size=1, buff_size=1024*1024)
            self.semantic_mask_publisher = rospy.Publisher(
                semantic_mask_topic_1, Image, latch=True, queue_size=1)
        else:
            self.image_subscriber_1 = message_filters.Subscriber(
                image_topic_1, CompressedImage)
            self.image_subscriber_2 = message_filters.Subscriber(
                image_topic_2, CompressedImage)
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.image_subscriber_1, self.image_subscriber_2], 100, 0.1)
            self.ts.registerCallback(self.image_callback_two)
            self.semantic_mask_publisher_1 = rospy.Publisher(
                semantic_mask_topic_1, Image, latch=True, queue_size=1)
            self.semantic_mask_publisher_2 = rospy.Publisher(
                semantic_mask_topic_2, Image, latch=True, queue_size=1)

    def image_callback(self, msg: CompressedImage):
        if abs(msg.header.stamp - self.last_processed_stamp) < self.min_time_passed:
            return
        self.last_processed_stamp = msg.header.stamp

        # Create semantic mask from image
        image = np.array(self.bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding='passthrough'))
        image_with_semantic = self.semantic_predictor(
            image, self.target_classes)

        # Publish image with highlighted semantic
        semantic_msg = self.bridge.cv2_to_imgmsg(
            image_with_semantic, encoding="passthrough")
        semantic_msg.header = msg.header
        semantic_msg.encoding = "rgb8"
        self.semantic_mask_publisher.publish(semantic_msg)

    def image_callback_two(self, msg_1: CompressedImage, msg_2: CompressedImage):
        if abs(msg_1.header.stamp - self.last_processed_stamp) < self.min_time_passed:
            return
        self.last_processed_stamp = msg_1.header.stamp

        # Create semantic mask from image
        image_1 = np.array(self.bridge.compressed_imgmsg_to_cv2(msg_1))
        image_with_semantic_1 = self.semantic_predictor(
            image_1, self.target_classes)
        image_2 = np.array(self.bridge.compressed_imgmsg_to_cv2(msg_2))
        image_with_semantic_2 = self.semantic_predictor(
            image_2, self.target_classes)

        # Publish image with highlighted semantic
        semantic_msg_1 = self.bridge.cv2_to_imgmsg(image_with_semantic_1)
        semantic_msg_1.header = msg_1.header
        semantic_msg_1.encoding = "rgb8"
        semantic_msg_2 = self.bridge.cv2_to_imgmsg(image_with_semantic_2)
        semantic_msg_2.header = msg_2.header
        semantic_msg_2.encoding = "rgb8"

        self.semantic_mask_publisher_1.publish(semantic_msg_1)
        self.semantic_mask_publisher_2.publish(semantic_msg_2)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    semantic_mapper = SemanticMapper()
    semantic_mapper.run()
