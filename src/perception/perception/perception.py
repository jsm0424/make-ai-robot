import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
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
        self.declare_parameter('model_path', 'path/to/your/best.pt') # Launch로 해서 상관 없음
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        
        # Load YOLO Model
        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            self.model = YOLO("yolov8n.pt") # Fallback to standard model if custom fails
            self.get_logger().warn("Custom model not found, using standard YOLOv8n")

        self.bridge = CvBridge()
        
        # Define target classes (What counts as "edible"?)
        # Update these names to match your labeled dataset classes
        self.target_classes = ['nurse', 'red_cone', 'green_cone', 'blue_cone', 'badapple','freshapple','badbanana','freshbanana','badpizza','freshpizza'] 

        # 2. Publishers
        self.pub_detection_img = self.create_publisher(Image, '/camera/detections/image', 10)
        self.pub_labels = self.create_publisher(String, '/detections/labels', 10)
        self.pub_distance = self.create_publisher(Float32, '/detections/distance', 10)
        self.pub_speech = self.create_publisher(String, '/robot_dog/speech', 10)

        # 3. Subscribers (Synchronized)
        # We use message_filters to ensure RGB and Depth align in time
        self.sub_rgb = Subscriber(self, Image, '/camera_top/image', qos_profile=qos_profile_sensor_data)
        self.sub_depth = Subscriber(self, Image, '/camera_top/depth', qos_profile=qos_profile_sensor_data)
        
        # Sync allows a small time roughly 0.1s difference
        self.ts = ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)
        
        self.get_logger().info('Perception Node Started')

    def image_callback(self, rgb_msg, depth_msg):
        try:
            # Convert ROS images to OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            # Depth is usually 32-bit float (meters) or 16-bit uint (mm)
            # Assuming simulation outputs 32FC1 (meters) for this example
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        height, width, _ = cv_img.shape
        
        # Run Inference
        results = self.model(cv_img, verbose=False)
        
        detected_label = "None"
        detected_dist = -1.0
        speech_cmd = "None"
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 1. Get Bounding Box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])

                if conf < 0.5: continue # Confidence threshold

                # 2. Calculate Center of Box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # 3. Get Distance from Depth Image
                # Safety check for boundaries
                center_x = max(0, min(center_x, width - 1))
                center_y = max(0, min(center_y, height - 1))
                
                # Get depth value at center pixel
                dist = depth_img[center_y, center_x]
                
                # Handle invalid depth (NaN or Inf)
                if np.isnan(dist) or np.isinf(dist):
                    dist = 0.0

                # 4. Logic: Is it Edible?
                if cls_name in self.target_classes:
                    detected_label = cls_name
                    detected_dist = float(dist)

                    # 5. Logic: Center Region Rule
                    # Exclude leftmost 1/5 and rightmost 1/5
                    # Region is [0.2*W, 0.8*W]
                    boundary_left = width * 0.2
                    boundary_right = width * 0.8

                    if boundary_left < center_x < boundary_right:
                        speech_cmd = "bark"
                    else:
                        speech_cmd = "None"

                    # Visual feedback: Draw box
                    cv2.rectangle(cv_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(cv_img, f"{cls_name} {dist:.2f}m", (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # We only prioritize the first valid edible object found for speech
                    break 
        cv2.imshow("YOLO Perception", cv_img)
        cv2.waitKey(1)
        # Publish Outputs
        self.pub_detection_img.publish(self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8'))
        self.pub_labels.publish(String(data=detected_label))
        self.pub_distance.publish(Float32(data=detected_dist))
        self.pub_speech.publish(String(data=speech_cmd))

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()