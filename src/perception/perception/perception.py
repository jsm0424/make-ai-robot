import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer
from ultralytics import YOLO

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # 1. Configuration
        # Default to a standard path, but this should be set via launch file usually
        self.declare_parameter('model_path', 'src/yolov8n.pt') 
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        
        # Load YOLO Model
        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            self.get_logger().warn(f"Custom model not found ({e}), loading standard YOLOv8n as fallback...")
            self.model = YOLO("yolov8n.pt") 

        self.bridge = CvBridge()
        
        # [CRITICAL] Class Definitions
        # p=pizza, a=apple, b=banana. good=edible, bad=inedible.
        # We also include common objects (nurse, cone, box) that might be in the bag file.
        self.target_classes = [
            'agood', 'abad', 'bgood', 'bbad', 'pgood', 'pbad', 
            'nurse', 'cone', 'sign', 'box'
        ]
        
        # Only these trigger the "bark" command
        self.edible_classes = ['agood', 'bgood', 'pgood'] 

        # 2. Publishers
        self.pub_detection_img = self.create_publisher(Image, '/camera/detections/image', 10)
        self.pub_labels = self.create_publisher(String, '/detections/labels', 10)
        self.pub_distance = self.create_publisher(Float32, '/detections/distance', 10)
        self.pub_speech = self.create_publisher(String, '/robot_dog/speech', 10)

        # 3. Subscribers
        # We need to sync RGB and Depth to get distance for the specific pixel
        self.sub_rgb = Subscriber(self, Image, '/camera_top/image', qos_profile=qos_profile_sensor_data)
        self.sub_depth = Subscriber(self, Image, '/camera_top/depth', qos_profile=qos_profile_sensor_data)
        
        # ApproximateTimeSynchronizer: Allows slight timestamp mismatch (slop=0.1s)
        self.ts = ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        # Requirement: Must subscribe to these, even if unused for logic
        self.create_subscription(PointCloud2, '/camera_top/points', self.points_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera_top/camera_info', self.info_callback, qos_profile_sensor_data)
        
        self.get_logger().info('Perception Node Started')

    def points_callback(self, msg):
        pass # Placeholder to satisfy requirements

    def info_callback(self, msg):
        pass # Placeholder to satisfy requirements

    def image_callback(self, rgb_msg, depth_msg):
        try:
            # Convert ROS images to OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            # Depth is 32FC1 (meters)
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        height, width, _ = cv_img.shape
        
        # Run YOLO Inference
        results = self.model(cv_img, verbose=False)
        
        # Default States
        current_frame_label = "None"
        current_frame_dist = 0.0
        speech_cmd = "None"
        
        # Define Regions (Left 1/5, Middle 3/5, Right 1/5)
        boundary_left = int(width * 0.2)
        boundary_right = int(width * 0.8)

        # Draw Region Lines (Yellow, matching reference)
        cv2.line(cv_img, (boundary_left, 0), (boundary_left, height), (0, 255, 255), 2)
        cv2.line(cv_img, (boundary_right, 0), (boundary_right, height), (0, 255, 255), 2)
        
        # Priority Logic: If multiple objects, prioritizing the "Barking" condition
        detected_edible_in_center = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 1. Get Class and Confidence
                cls_id = int(box.cls[0])
                if cls_id >= len(self.model.names): continue # Safety check
                
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])

                if conf < 0.5: continue 
                if cls_name not in self.target_classes: continue

                # 2. Get Bounding Box Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # 3. Calculate Center
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # 4. Get Distance from Depth Map
                # Clip coordinates to be safe
                safe_x = max(0, min(center_x, width - 1))
                safe_y = max(0, min(center_y, height - 1))
                
                raw_dist = depth_img[safe_y, safe_x]
                
                # Handle NaN/Inf values from depth camera
                if np.isnan(raw_dist) or np.isinf(raw_dist):
                    dist = 0.0
                else:
                    dist = float(raw_dist)

                # 5. Logic for Barking (Edible + Center + Close)
                is_edible = cls_name in self.edible_classes
                is_centered = boundary_left < center_x < boundary_right
                is_close = dist < 3.0 and dist > 0.1 # Min dist to avoid noise

                # Visualization Color: Red if it triggers bark, Green otherwise
                box_color = (0, 255, 0) 
                
                if is_edible:
                    # Update status for publishing
                    # We overwrite this if we find an edible object, so the label reflects the food
                    current_frame_label = cls_name
                    current_frame_dist = dist
                    
                    if is_centered and is_close:
                        speech_cmd = "bark"
                        detected_edible_in_center = True
                        box_color = (0, 0, 255) # Red box for active target
                        
                        # Visual marker for center lock
                        cv2.circle(cv_img, (center_x, center_y), 5, (0, 0, 255), -1)

                # Draw Bounding Box & Text
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), box_color, 2)
                
                # Label Text: "pgood 2.5m"
                label_text = f"{cls_name} {dist:.2f}m"
                cv2.putText(cv_img, label_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # 6. Publish Results
        # If we didn't find an edible object in center, we might still want to publish 
        # the label of whatever we DID find (like a nurse or bad food), 
        # but 'speech' must remain "None".
        
        # Publish Image
        out_img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        self.pub_detection_img.publish(out_img_msg)
        
        # Publish Data
        self.pub_labels.publish(String(data=current_frame_label))
        self.pub_distance.publish(Float32(data=current_frame_dist))
        self.pub_speech.publish(String(data=speech_cmd))

        # Optional: Local Debug View (Disable on robot if headless)
        cv2.imshow("Perception View", cv_img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()