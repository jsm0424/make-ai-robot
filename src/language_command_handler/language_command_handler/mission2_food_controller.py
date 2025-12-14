#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math
import time
import os
import numpy as np 
from ament_index_python.packages import get_package_share_directory

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics library not found.")

class Mission2FoodController(Node):
    def __init__(self):
        super().__init__('mission2_food_controller')

        # === [Coordinate Definitions] ===
        # Strategy 1: Small rooms first (2-3-4-8-5-6-16-15-13-14)
        raw_strat_1 = [
            (-2.03, -15.32, 0.34), # Room 2
            (2.74, -15.30, 0.34),  # Room 3
            (2.79, -8.65, 0.34),   # Room 4
            (-3.79, -9.45, 0.34),  # Room 8
            (-7.79, 0.94, 0.34),   # Room 5
            (7.81, 0.85, 0.34),    # Room 6
            (-8.22, 4.75, 0.32),   # Room 16
            (-8.73, 11.20, 0.34),  # Room 15
            (8.73, 11.20, 0.34),   # Room 13
            (8.22, 4.75, 0.32),    # Room 14
        ]

        # Strategy 2: Closest/Large rooms first (12-1-9-7-10-11)
        raw_strat_2 = [
            (-7.60, -26.70, 0.34),
            (-7.71, -26.06, 0.33),


            # Room 1 (간호사 방 옆)
            (-5.73, -23.17, 0.34),
            (-7.49, -22.97, 0.34),
            (-7.91, -19.64, 0.34),
            (-8.55, -16.80, 0.34),


            # Room 9 (large)
            (-5.89, -9.75, 0.34),
            (-7.05, -8.96, 0.34),
            (-7.15, -8.93, 0.34),
            (-7.71, -8.77, 0.34),
            (-7.79, -6.46, 0.34),
            (-7.37, -2.51, 0.34),
            (-7.71, -5.95, 0.34),


            # Room 7
            (5.55, -8.97, 0.33),
            (6.88, -8.87, 0.34),
            (7.79, -8.09, 0.34),


            # Room 10 (large)
            (5.08, -23.15, 0.34),
            (8.61, -21.03, 0.37),
            (9.07, -20.75, 0.34),
            (9.08, -17.74, 0.34),


            # Room 11 (large)
            (5.51, -25.14, 0.33),
            (6.76, -26.66, 0.34),
            (6.17, -26.55, 0.33),
            (4.74, -27.36, 0.34),
            (7.97, -27.44, 0.33),
            (-1.61, -27.10, 0.33),
        ]

        # [Logic] Filter Duplicates & Set Order
        # We start with Strategy 2. If that fails, we append Strategy 1.
        self.strat_2_coords = self.filter_coordinates(raw_strat_2)
        self.strat_1_coords = self.filter_coordinates(raw_strat_1)
        
        # Initial Search Queue is Strategy 2
        self.SEARCH_QUEUE = self.strat_2_coords
        self.current_strategy = 2 # Track which strategy we are running
        
        self.get_logger().info(f"Loaded {len(self.SEARCH_QUEUE)} locations for Strategy 2.")

        # === [Settings] ===
        self.EDIBLE_FOODS = ['agood', 'bgood', 'pgood']
        self.INEDIBLE_FOODS = ['abad', 'bbad', 'pbad']
        self.ALL_FOODS = self.EDIBLE_FOODS + self.INEDIBLE_FOODS

        self.APPROACH_DIST = 0.6      
        self.CENTER_TOLERANCE = 15    
        self.MAX_ROT_SPEED = 0.3
        self.P_GAIN = 0.002           
        self.CONFIDENCE_THRESHOLD = 0.7 # [REQ] High confidence only

        # === [Model] ===
        try:
            pkg_share = get_package_share_directory('language_command_handler')
            default_model_path = os.path.join(pkg_share, 'models', 'food_model.pt')
        except:
            default_model_path = 'src/language_command_handler/models/food_model.pt'
            
        self.declare_parameter('model_path', default_model_path)
        self.model = YOLO(self.get_parameter('model_path').value)

        # === [Comms] ===
        self.nav_pub = self.create_publisher(PoseStamped, '/navigator/input_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/robot_dog/speech', 10)
        self.nav_sub = self.create_subscription(String, '/navigator/status', self.status_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/go1_pose', self.pose_callback, 10)

        qos_policy = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.img_sub = self.create_subscription(Image, '/camera_face/image', self.img_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera_face/depth', self.depth_callback, qos_policy)

        self.bridge = CvBridge()
        self.cv_image = None
        self.latest_depth_image = None
        self.robot_pose = None

        # State Machine
        self.state = 0
        self.location_idx = 0
        self.search_start_time = 0.0
        self.SEARCH_TIMEOUT = 10.0 
        self.wait_until_time = 0.0
        self.is_navigating = False
        self.target_class_name = None 

        self.create_timer(0.1, self.mission_loop)
        self.get_logger().info("🍔 Mission 2: Strategy 2 (Large Rooms) Started")

    def filter_coordinates(self, coords_list):
        """ Removes duplicates that are too close (< 0.5m) """
        filtered = []
        for c in coords_list:
            is_duplicate = False
            for existing in filtered:
                dist = math.hypot(c[0] - existing[0], c[1] - existing[1])
                if dist < 0.5: # 0.5m threshold
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append({'x': c[0], 'y': c[1], 'yaw': c[2]})
        return filtered

    def pose_callback(self, msg): self.robot_pose = msg
    def depth_callback(self, msg):
        try: self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except: pass
    def img_callback(self, msg):
        try: self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass

    def status_callback(self, msg):
        if msg.data == "ARRIVED":
            self.is_navigating = False
            if self.state == 1:
                self.get_logger().info("📍 Arrived. Scanning...")
                self.set_sleep(1.0)
                self.search_start_time = self.get_clock().now().nanoseconds / 1e9
                self.state = 2

    def set_sleep(self, seconds):
        self.wait_until_time = self.get_clock().now().nanoseconds / 1e9 + seconds
        self.cmd_vel_pub.publish(Twist())

    def is_sleeping(self):
        return (self.get_clock().now().nanoseconds / 1e9) < self.wait_until_time

    def send_nav_command(self, pose_dict):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(pose_dict['x'])
        msg.pose.position.y = float(pose_dict['y'])
        yaw = float(pose_dict['yaw'])
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.nav_pub.publish(msg)
        self.is_navigating = True

    def get_depth_dist(self, cx, cy):
        if self.latest_depth_image is None: return 99.9
        h, w = self.latest_depth_image.shape
        cx = int(np.clip(cx, 0, w-1))
        cy = int(np.clip(cy, 0, h-1))
        try:
            val = self.latest_depth_image[cy, cx]
            if np.isnan(val) or np.isinf(val) or val == 0: return 99.9
            return float(val)
        except: return 99.9

    # === [REQ] Highest Confidence Logic ===
    def process_food_detection(self):
        if self.cv_image is None or self.model is None: return False, 0.0, 99.9, None

        results = self.model(self.cv_image, verbose=False, conf=self.CONFIDENCE_THRESHOLD)
        _, img_w, _ = self.cv_image.shape
        center_x_screen = img_w / 2

        best_box = None
        max_conf = -1.0 # Start lower than threshold
        detected_class = None

        # Iterate ALL detections to find the absolute best one
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                
                if cls_name in self.ALL_FOODS:
                    # [REQ] Strict check > 0.7 and > current max
                    if conf > self.CONFIDENCE_THRESHOLD and conf > max_conf:
                        max_conf = conf
                        best_box = box
                        detected_class = cls_name
        
        if best_box:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            dist = self.get_depth_dist(cx, cy)
            error_x = center_x_screen - cx

            # Visualization
            color = (0, 255, 0) if detected_class in self.EDIBLE_FOODS else (0, 0, 255)
            cv2.rectangle(self.cv_image, (x1,y1), (x2,y2), color, 2)
            cv2.putText(self.cv_image, f"{detected_class} {max_conf:.2f} {dist:.2f}m", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow("Food Search", self.cv_image)
            cv2.waitKey(1)
            
            return True, error_x, dist, detected_class

        cv2.imshow("Food Search", self.cv_image)
        cv2.waitKey(1)
        return False, 0.0, 99.9, None

    def mission_loop(self):
        if self.robot_pose is None: return
        if self.is_sleeping(): return

        # [State 0] Move to Next Search Location
        if self.state == 0:
            if self.location_idx < len(self.SEARCH_QUEUE):
                target = self.SEARCH_QUEUE[self.location_idx]
                self.get_logger().info(f"🚗 Going to Spot {self.location_idx+1}/{len(self.SEARCH_QUEUE)} (Strat {self.current_strategy})")
                self.send_nav_command(target)
                self.state = 1
            else:
                # [REQ] Strategy Switch Logic
                if self.current_strategy == 2:
                    self.get_logger().warn("⚠️ Strategy 2 finished. Switching to Strategy 1 (Small Rooms)...")
                    self.SEARCH_QUEUE = self.strat_1_coords # Load Strat 1
                    self.location_idx = 0 # Reset Index
                    self.current_strategy = 1
                    self.state = 0 # Restart loop with new queue
                else:
                    self.get_logger().info("🏁 All Strategies exhausted. Mission End.")
                    self.state = 5

        elif self.state == 1: pass 

        # [State 2] Scan & Detect
        elif self.state == 2:
            detected, error_x, dist, cls_name = self.process_food_detection()
            
            elapsed = (self.get_clock().now().nanoseconds / 1e9) - self.search_start_time
            if elapsed > self.SEARCH_TIMEOUT:
                self.get_logger().warn("❌ Nothing found here. Moving to next.")
                self.cmd_vel_pub.publish(Twist()) 
                self.location_idx += 1
                self.state = 0
                return

            if detected:
                self.target_class_name = cls_name
                if abs(error_x) < self.CENTER_TOLERANCE:
                    self.cmd_vel_pub.publish(Twist()) 
                    self.get_logger().info(f"🎯 Locked: {cls_name} at {dist:.2f}m")
                    self.state = 3
                else:
                    raw_z = error_x * self.P_GAIN
                    if raw_z > 0: ang_z = min(raw_z, self.MAX_ROT_SPEED)
                    else: ang_z = max(raw_z, -self.MAX_ROT_SPEED)
                    cmd = Twist()
                    cmd.angular.z = float(ang_z)
                    self.cmd_vel_pub.publish(cmd)
            else:
                cmd = Twist()
                cmd.angular.z = 0.2
                self.cmd_vel_pub.publish(cmd)

        # [State 3] Approach
        elif self.state == 3:
            detected, error_x, dist, cls_name = self.process_food_detection()

            if not detected:
                self.get_logger().warn("⚠️ Lost target! Re-scanning.")
                self.cmd_vel_pub.publish(Twist())
                self.state = 2
                return

            cmd = Twist()
            ang_z = error_x * self.P_GAIN
            cmd.angular.z = float(np.clip(ang_z, -0.3, 0.3))

            if dist > self.APPROACH_DIST:
                cmd.linear.x = 0.3 
            else:
                cmd.linear.x = 0.0 
                self.cmd_vel_pub.publish(cmd)
                self.get_logger().info("✅ Reached Food!")
                self.set_sleep(1.0)
                self.state = 4
                return
            
            self.cmd_vel_pub.publish(cmd)

        # [State 4] Decision
        elif self.state == 4:
            if self.target_class_name in self.EDIBLE_FOODS:
                self.get_logger().info(f"😋 Found EDIBLE ({self.target_class_name})! Barking...")
                msg = String()
                msg.data = "bark"
                self.speech_pub.publish(msg) 
                self.set_sleep(2.0) 
            else:
                self.get_logger().info(f"🤢 Found INEDIBLE ({self.target_class_name}). Ignoring.")
                self.set_sleep(1.0)

            self.location_idx += 1
            self.state = 0

        elif self.state == 5: pass 

def main(args=None):
    rclpy.init(args=args)
    node = Mission2FoodController()
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