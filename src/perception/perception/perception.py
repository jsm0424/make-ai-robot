#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import message_filters
import os
from ament_index_python.packages import get_package_share_directory

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # 1. Initialize CV Bridge and Parameters
        self.bridge = CvBridge()
        
        # Define the "Bad" labels that stop publication
        self.bad_labels = ['bbad', 'abad', 'pbad']
        # Define "Good" labels (optional list, but we rely on "not bad" logic here)
        self.target_labels = ['agood', 'pgood', 'bgood']  
        
        # 2. Load the YOLO Model
        try:
            package_share_directory = get_package_share_directory('perception')
            model_path = os.path.join(package_share_directory, 'models', 'food_model.pt')
            self.get_logger().info(f'Loading YOLO model from: {model_path}')
            self.model = YOLO(model_path)
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}. Check path.')

        # 3. Subscribers (Synchronized RGB + Depth)
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera_top/image')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera_top/depth')

        # Use ApproximateTimeSynchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        # 4. Publishers
        self.pub_debug_image = self.create_publisher(Image, '/camera/detections/image', 10)
        self.pub_labels = self.create_publisher(String, '/detections/labels', 10)
        self.pub_distance = self.create_publisher(Float32, '/detections/distance', 10)
        self.pub_speech = self.create_publisher(String, '/robot_dog/speech', 10)

        self.get_logger().info('Perception Node Initialized (High Conf Filter Mode)')

    def image_callback(self, rgb_msg, depth_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        height, width, _ = cv_image.shape
        
        # --- 1. Run Object Detection ---
        results = self.model(cv_image, verbose=False)
        
        # --- 2. Find Highest Confidence Object > 0.8 ---
        best_box = None
        max_conf = 0.8  # Start threshold at 0.8
        
        # Iterate through all results to find the single best box
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    best_box = box

        # If no object was found with conf > 0.8, we publish nothing and return
        if best_box is None:
            return

        # --- 3. Check Label ---
        cls_id = int(best_box.cls[0])
        label = self.model.names[cls_id]

        # STRICT FILTER: If the best label is "bad", publish NO topics.
        if label in self.bad_labels:
            return

        # --- 4. Process Valid Detection ---
        # If we are here, we have a "Good" object with Conf > 0.8
        
        # Extract Box Coordinates
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw Bounding Box (Green)
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw Label Text
        current_dist = 0.0
        # Depth Calculation
        if 0 <= cx < width and 0 <= cy < height:
            raw_depth = cv_depth[cy, cx]
            if isinstance(raw_depth, (float, np.float32, np.float64)):
                current_dist = float(raw_depth)
            else:
                current_dist = float(raw_depth) / 1000.0 # Handle mm to m conversion if needed

        label_text = f"{label} {max_conf:.2f} {current_dist:.2f}m"
        cv2.putText(cv_image, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(cv_image, (cx, cy), 5, (0, 0, 255), -1)

        # Logic for Speech (Bark if centered)
        left_limit = int(width * 0.2)
        right_limit = int(width * 0.8)
        
        # Draw region lines for visualization
        cv2.line(cv_image, (left_limit, 0), (left_limit, height), (0, 255, 255), 1)
        cv2.line(cv_image, (right_limit, 0), (right_limit, height), (0, 255, 255), 1)

        speech_cmd = "None"
        if left_limit <= cx <= right_limit:
            speech_cmd = "bark"

        # --- 5. Publish Topics ---
        
        # Topic 1: Image with Bounding Box
        out_img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        out_img_msg.header = rgb_msg.header
        self.pub_debug_image.publish(out_img_msg)

        # Topic 2: Label
        self.pub_labels.publish(String(data=label))

        # Topic 3: Distance
        self.pub_distance.publish(Float32(data=current_dist))

        # Topic 4: Speech
        self.pub_speech.publish(String(data=speech_cmd))

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()