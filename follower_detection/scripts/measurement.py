#!/usr/bin/env python3

import os
import rospy
import numpy as np
import ros_numpy
import torch
import tf
import cv2

from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge

from ultralytics import YOLO

class MeasurePersonPosition:
    def __init__(self):
        self.pcl_rgb_topic = '/tulio/camera/depth/color/points'
        self.this_directory = os.path.dirname(os.path.realpath(__file__))


        torch.cuda.set_device(0) # Set device to gpu
        self.model = YOLO(f'{self.this_directory}/yolov8n-pose.pt') # YOLO model
        self.model.to('cuda')
        self.bridge = CvBridge() # CV bridge

        self.target_id = 1
        # Subscribers
        rospy.Subscriber(self.pcl_rgb_topic, PointCloud2, self.pcl_callback)
        rospy.loginfo('Subscribed to %s', self.pcl_rgb_topic)

    def pcl_callback(self, msg):
        self.points_data = ros_numpy.numpify(msg)
        self.image_data = np.ascontiguousarray(self.points_data['rgb'].view((np.uint8, 4))[..., [0,1,2]])
        self.resolution = self.image_data.shape[:2]


        if not self.image_data is None:
            results = self.model.track(self.image_data, persist=True, classes=[0], tracker=f"{os.path.dirname(self.this_directory)}/config/tracker.yaml", verbose=False)
        
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        keypoints = results[0].keypoints.xyn.cpu().tolist()

        annotated_frame = results[0].plot()

        cv2.imshow('image', self.image_data)
        cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('measure_person_position')
    MeasurePersonPosition()
    rospy.spin()