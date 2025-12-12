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
        # This function tries 3 different ways to find your file.
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

        # --- 3. Publishers ---
        self.pub_detection_img = self.create_publisher(Image, '/camera/detections/image', 10)
        self.pub_labels = self.create_publisher(String, '/detections/labels', 10)
        self.pub_distance = self.create_publisher(Float32, '/detections/distance', 10)
        self.pub_speech = self.create_publisher(String, '/robot_dog/speech', 10)

        # --- 4. Subscribers ---
        self.create_subscription(Image, '/camera_top/depth', self.depth_callback, qos_policy)
        self.create_subscription(Image, '/camera_top/image', self.rgb_callback, qos_policy)
        self.create_subscription(PointCloud2, '/camera_top/points', lambda msg: None, qos_policy)
        self.create_subscription(CameraInfo, '/camera_top/camera_info', lambda msg: None, qos_policy)
        
        self.get_logger().info('Perception Node Started (Robust Path Finding)')

    def find_model_path(self, filename):
        """
        Tries to find the model file in multiple locations.
        """
        # Option A: The standard ROS2 install location (requires setup.py data_files)
        try:
            pkg_share = get_package_share_directory('perception')
            install_path = os.path.join(pkg_share, 'models', filename)
            if os.path.exists(install_path):
                return install_path
        except Exception:
            pass

        # Option B: Relative path from workspace root (common during development)
        # Assumes you run ros2 run from the folder 'make-ai-robot'
        dev_path = os.path.join(os.getcwd(), 'src', 'perception', 'perception', 'models', filename)
        if os.path.exists(dev_path):
            return dev_path
        
        # Option C: Another common relative path variation
        dev_path_2 = os.path.join(os.getcwd(), 'src', 'perception', 'models', filename)
        if os.path.exists(dev_path_2):
            return dev_path_2

        # Option D: Absolute path hardcoded (Update this if needed!)
        # This is the specific path you mentioned in your prompt
        abs_path = os.path.expanduser(f'~/make-ai-robot/src/perception/perception/models/{filename}')
        if os.path.exists(abs_path):
            return abs_path

        self.get_logger().warn(f"Model file '{filename}' not found in any standard location.")
        return filename # Return filename hoping YOLO finds it in current dir

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
            
            current_frame_label = "None"
            current_frame_dist = 0.0
            speech_cmd = "None"
            
            boundary_left = int(width * 0.2)
            boundary_right = int(width * 0.8)
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

                    dist = 0.0
                    if self.latest_depth_img is not None:
                        safe_x = np.clip(center_x, 0, width - 1)
                        safe_y = np.clip(center_y, 0, height - 1)
                        try:
                            raw_dist = self.latest_depth_img[safe_y, safe_x]
                            if not (np.isnan(raw_dist) or np.isinf(raw_dist)):
                                dist = float(raw_dist)
                        except IndexError: pass

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

            # --- DEBUGGING OVERLAY (LABELS) ---
            # 화면 상단에 상태 정보를 보여주는 검은색 바 추가
            overlay_h = 60
            cv2.rectangle(cv_img, (0, 0), (width, overlay_h), (0, 0, 0), -1)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 왼쪽: 감지된 객체 및 거리
            cv2.putText(cv_img, f"OBJ: {current_frame_label}", (10, 25), font, 0.6, (255, 255, 255), 1)
            cv2.putText(cv_img, f"DST: {current_frame_dist:.2f}m", (10, 50), font, 0.6, (255, 255, 255), 1)
            
            # 오른쪽: 명령 상태
            cmd_color = (0, 255, 0) if speech_cmd == "bark" else (100, 100, 100)
            cv2.putText(cv_img, f"CMD: {speech_cmd}", (width - 160, 40), font, 0.7, cmd_color, 2)

            # --- PUBLISH & SHOW ---
            self.pub_detection_img.publish(self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8'))
            self.pub_labels.publish(String(data=current_frame_label))
            self.pub_distance.publish(Float32(data=current_frame_dist))
            self.pub_speech.publish(String(data=speech_cmd))

            # 로컬 화면 띄우기 (imshow + waitKey)
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
        cv2.destroyAllWindows() # 종료 시 윈도우 닫기 추가
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()