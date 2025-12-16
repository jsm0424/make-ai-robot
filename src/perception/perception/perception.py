import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # --- 1. Robust Model Loading ---
        final_model_path = self.find_model_path('food_model.pt')
        
        self.declare_parameter('model_path', final_model_path)
        self.declare_parameter('debug_mode', False)
        self.debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value

        self.get_logger().info(f'Loading YOLO model from: {final_model_path}')
        
        try:
            self.model = YOLO(final_model_path)
        except Exception as e:
            self.get_logger().error(f"CRITICAL: Failed to load model at {final_model_path}. Error: {e}")
            self.get_logger().warn("Downloading standard YOLOv8n.pt as emergency fallback...")
            self.model = YOLO("yolov8n.pt") 

        self.bridge = CvBridge()
        self.latest_depth_img = None 
        self.is_processing = False

        # "edible_classes" kept for reference, but logic now applies to ALL target_classes below
        self.edible_classes = ['agood', 'bgood', 'pgood'] 
        self.target_classes = [
            'agood', 'abad', 'bgood', 'bbad', 'pgood', 'pbad', 
            'nurse', 'cone', 'sign', 'box'
        ]

        # --- 2. QoS Profile ---
        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- 3. Publishers (Strictly adhering to Table) ---
        # /camera/detections/image | sensor_msgs/Image
        self.pub_detection_img = self.create_publisher(Image, '/camera/detections/image', 10)
        
        # /detections/labels | std_msgs/String
        self.pub_labels = self.create_publisher(String, '/detections/labels', 10)
        
        # /detections/distance | std_msgs/Float32
        self.pub_distance = self.create_publisher(Float32, '/detections/distance', 10)
        
        # /robot_dog/speech | std_msgs/String
        self.pub_speech = self.create_publisher(String, '/robot_dog/speech', 10)

        # --- 4. Subscribers ---
        self.create_subscription(Image, '/camera_top/depth', self.depth_callback, qos_policy)
        self.create_subscription(Image, '/camera_top/image', self.rgb_callback, qos_policy)
        self.create_subscription(PointCloud2, '/camera_top/points', lambda msg: None, qos_policy)
        self.create_subscription(CameraInfo, '/camera_top/camera_info', lambda msg: None, qos_policy)
        
        self.get_logger().info('Perception Node Started (Topics Verified)')

    def find_model_path(self, filename):
        """
        Tries to find the model file in multiple locations.
        """
        try:
            pkg_share = get_package_share_directory('perception')
            install_path = os.path.join(pkg_share, 'models', filename)
            if os.path.exists(install_path):
                return install_path
        except Exception:
            pass

        dev_path = os.path.join(os.getcwd(), 'src', 'perception', 'perception', 'models', filename)
        if os.path.exists(dev_path):
            return dev_path
        
        dev_path_2 = os.path.join(os.getcwd(), 'src', 'perception', 'models', filename)
        if os.path.exists(dev_path_2):
            return dev_path_2

        abs_path = os.path.expanduser(f'~/make-ai-robot/src/perception/perception/models/{filename}')
        if os.path.exists(abs_path):
            return abs_path

        self.get_logger().warn(f"Model file '{filename}' not found in any standard location.")
        return filename 

    def depth_callback(self, msg):
        try:
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception:
            pass

    def rgb_callback(self, msg):
        if self.is_processing: return
        self.is_processing = True
        
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width, _ = cv_img.shape
            
            # Inference
            results = self.model(cv_img, verbose=False, imgsz=320, conf=0.5)
            
            # Default values to publish if nothing detected
            current_frame_label = "None"
            current_frame_dist = 0.0
            speech_cmd = "None"
            
            # Visual boundaries (Optional visual aid)
            boundary_left = int(width * 0.2)
            boundary_right = int(width * 0.8)
            cv2.line(cv_img, (boundary_left, 0), (boundary_left, height), (0, 255, 255), 2)
            cv2.line(cv_img, (boundary_right, 0), (boundary_right, height), (0, 255, 255), 2)

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0]) # Confidence score

                    if cls_id >= len(self.model.names): continue
                    cls_name = self.model.names[cls_id]
                    
                    # 1. Check Class validity
                    if cls_name not in self.target_classes: continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # 2. Get Distance
                    dist = 0.0
                    if self.latest_depth_img is not None:
                        safe_x = np.clip(center_x, 0, width - 1)
                        safe_y = np.clip(center_y, 0, height - 1)
                        try:
                            raw_dist = self.latest_depth_img[safe_y, safe_x]
                            if not (np.isnan(raw_dist) or np.isinf(raw_dist)):
                                dist = float(raw_dist)
                        except IndexError: pass

                    # 3. Logic: Bark if Conf > 0.7 AND Dist <= 3.0m
                    is_confident = conf > 0.7
                    is_near = 0.0 < dist <= 3.0

                    box_color = (0, 255, 0) # Green for detection

                    # If this object meets the criteria, it becomes the "active" detection for publishing
                    if is_confident and is_near:
                        current_frame_label = cls_name
                        current_frame_dist = dist
                        speech_cmd = "bark"
                        box_color = (0, 0, 255) # Red for active trigger
                        cv2.circle(cv_img, (center_x, center_y), 5, (0, 0, 255), -1)

                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(cv_img, f"{cls_name} {dist:.1f}m ({conf:.2f})", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # --- DEBUGGING OVERLAY ---
            overlay_h = 60
            cv2.rectangle(cv_img, (0, 0), (width, overlay_h), (0, 0, 0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Left: Detected Object & Distance
            cv2.putText(cv_img, f"OBJ: {current_frame_label}", (10, 25), font, 0.6, (255, 255, 255), 1)
            cv2.putText(cv_img, f"DST: {current_frame_dist:.2f}m", (10, 50), font, 0.6, (255, 255, 255), 1)
            
            # Right: Command State
            cmd_color = (0, 255, 0) if speech_cmd == "bark" else (100, 100, 100)
            cv2.putText(cv_img, f"CMD: {speech_cmd}", (width - 160, 40), font, 0.7, cmd_color, 2)

            # --- PUBLISHING (Matches Table) ---
            self.pub_detection_img.publish(self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8'))
            self.pub_labels.publish(String(data=current_frame_label))
            self.pub_distance.publish(Float32(data=current_frame_dist))
            self.pub_speech.publish(String(data=speech_cmd))

            cv2.imshow("YOLO Debug View", cv_img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")
        finally:
            self.is_processing = False

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()