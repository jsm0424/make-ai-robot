import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # --- 1. Configuration ---
        self.declare_parameter('model_path', 'src/yolov8n.pt') 
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        
        self.declare_parameter('debug_mode', True)
        self.debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value

        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        try:
            self.model = YOLO(model_path)
        except Exception:
            self.get_logger().warn("Custom model not found, loading standard YOLOv8n...")
            self.model = YOLO("yolov8n.pt") 

        self.bridge = CvBridge()
        
        # State Variables
        self.latest_depth_img = None 
        self.is_processing = False  # <--- PREVENTS FREEZING

        self.edible_classes = ['agood', 'bgood', 'pgood'] 
        self.target_classes = [
            'agood', 'abad', 'bgood', 'bbad', 'pgood', 'pbad', 
            'nurse', 'cone', 'sign', 'box'
        ]

        # --- 2. QoS Profile (Best Effort is critical for video) ---
        qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- 3. Publishers ---
        self.pub_detection_img = self.create_publisher(Image, '/camera/detections/image', 10)
        self.pub_labels = self.create_publisher(String, '/detections/labels', 10)
        self.pub_distance = self.create_publisher(Float32, '/detections/distance', 10)
        self.pub_speech = self.create_publisher(String, '/robot_dog/speech', 10)

        # --- 4. Subscribers ---
        # Depth: Fast update
        self.create_subscription(Image, '/camera_top/depth', self.depth_callback, qos_policy)
        
        # RGB: Main trigger
        self.create_subscription(Image, '/camera_top/image', self.rgb_callback, qos_policy)

        # Dummy subs to satisfy requirements
        self.create_subscription(PointCloud2, '/camera_top/points', lambda msg: None, qos_policy)
        self.create_subscription(CameraInfo, '/camera_top/camera_info', lambda msg: None, qos_policy)
        
        self.get_logger().info('Perception Node Started (Stabilized)')

    def depth_callback(self, msg):
        try:
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception:
            pass

    def rgb_callback(self, msg):
        # [CRITICAL FIX] If we are already working, DROP this frame immediately.
        # This prevents the queue from building up and freezing the node.
        if self.is_processing:
            return

        self.is_processing = True
        
        try:
            # 1. Convert Image
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width, _ = cv_img.shape
            
            # 2. Inference (Keep it fast)
            results = self.model(cv_img, verbose=False, imgsz=320, conf=0.5)
            
            current_frame_label = "None"
            current_frame_dist = 0.0
            speech_cmd = "None"
            
            boundary_left = int(width * 0.2)
            boundary_right = int(width * 0.8)
            
            # Draw Regions
            cv2.line(cv_img, (boundary_left, 0), (boundary_left, height), (0, 255, 255), 2)
            cv2.line(cv_img, (boundary_right, 0), (boundary_right, height), (0, 255, 255), 2)

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id >= len(self.model.names): continue
                    cls_name = self.model.names[cls_id]
                    if cls_name not in self.target_classes: continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # 3. Distance Lookup (Safe)
                    dist = 0.0
                    if self.latest_depth_img is not None:
                        safe_x = np.clip(center_x, 0, width - 1)
                        safe_y = np.clip(center_y, 0, height - 1)
                        try:
                            raw_dist = self.latest_depth_img[safe_y, safe_x]
                            if not (np.isnan(raw_dist) or np.isinf(raw_dist)):
                                dist = float(raw_dist)
                        except IndexError:
                            pass

                    # Logic
                    is_edible = cls_name in self.edible_classes
                    is_centered = boundary_left < center_x < boundary_right
                    is_close = 0.1 < dist < 3.0

                    box_color = (0, 255, 0)
                    if is_edible:
                        current_frame_label = cls_name
                        current_frame_dist = dist
                        if is_centered and is_close:
                            speech_cmd = "bark"
                            box_color = (0, 0, 255)
                            cv2.circle(cv_img, (center_x, center_y), 5, (0, 0, 255), -1)

                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(cv_img, f"{cls_name} {dist:.1f}m", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Publish
            self.pub_detection_img.publish(self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8'))
            self.pub_labels.publish(String(data=current_frame_label))
            self.pub_distance.publish(Float32(data=current_frame_dist))
            self.pub_speech.publish(String(data=speech_cmd))

            if self.debug_mode:
                cv2.imshow("Perception Debug", cv_img)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")
        
        finally:
            # [CRITICAL] Always release the lock, even if code crashes
            self.is_processing = False

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