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

class Mission6NurseController(Node):
    def __init__(self):
        super().__init__('mission6_nurse_controller')

        # === [Settings] ===
        self.ROOM_ENTRANCE = {'x': -6.68, 'y': -24.92, 'yaw': -3.13}
        
        # [Precision Settings]
        self.CENTER_TOLERANCE = 10   
        self.P_GAIN = 0.0015         
        self.MAX_ROT_SPEED = 0.3     

        # Model Path
        try:
            pkg_share = get_package_share_directory('language_command_handler')
            default_model_path = os.path.join(pkg_share, 'models', 'nurse_model.pt')
        except:
            default_model_path = 'src/language_command_handler/models/nurse_model.pt'
            
        self.declare_parameter('model_path', default_model_path)
        self.model = YOLO(self.get_parameter('model_path').value)

        # Communication
        self.nav_pub = self.create_publisher(PoseStamped, '/navigator/input_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav_sub = self.create_subscription(String, '/navigator/status', self.status_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/go1_pose', self.pose_callback, 10)
        self.speech_pub = self.create_publisher(String, '/robot_dog/speech', 10)

        qos_policy = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.img_sub = self.create_subscription(Image, '/camera_face/image', self.img_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera_face/depth', self.depth_callback, qos_policy)

        self.bridge = CvBridge()
        self.cv_image = None
        self.latest_depth_image = None
        
        self.robot_pose = None
        self.rx = 0.0
        self.ry = 0.0
        self.ryaw = 0.0

        self.nurse_global_x = 0.0
        self.nurse_global_y = 0.0
        
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.state = 0
        self.start_delay = 0 
        self.wait_until_time = 0.0
        self.is_navigating = False

        self.create_timer(0.1, self.mission_loop)
        self.get_logger().info("🧠 Mission 6: Hardcoded Yaw Mode")

    def pose_callback(self, msg):
        self.robot_pose = msg
        self.rx = msg.pose.position.x
        self.ry = msg.pose.position.y
        q = msg.pose.orientation
        self.ryaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except: pass

    def img_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass

    def status_callback(self, msg):
        if msg.data == "ARRIVED":
            self.is_navigating = False
            if self.state == 1: 
                self.get_logger().info("🚪 Arrived. Stabilizing for 2s.")
                self.set_sleep(2.0)
                self.state = 2

    def set_sleep(self, seconds):
        self.wait_until_time = self.get_clock().now().nanoseconds / 1e9 + seconds
        self.cmd_vel_pub.publish(Twist())

    def is_sleeping(self):
        return (self.get_clock().now().nanoseconds / 1e9) < self.wait_until_time

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

    def process_centering(self):
        if self.cv_image is None or self.model is None: return False, 99.9

        results = self.model(self.cv_image, verbose=False, conf=0.5)
        _, img_w, _ = self.cv_image.shape
        center_x_screen = img_w / 2

        best_box = None
        max_conf = -1.0

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0: 
                    if float(box.conf[0]) > max_conf:
                        max_conf = float(box.conf[0])
                        best_box = box
        
        if best_box:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            dist = self.get_depth_dist(cx, cy)

            error_x = center_x_screen - cx
            
            # Draw Debug
            cv2.rectangle(self.cv_image, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.line(self.cv_image, (int(center_x_screen), 0), (int(center_x_screen), 1000), (0,0,255), 1)
            cv2.putText(self.cv_image, f"Err: {error_x:.1f}", (cx, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            cv2.imshow("Centering Nurse", self.cv_image)
            cv2.waitKey(1)

            if abs(error_x) < self.CENTER_TOLERANCE:
                self.cmd_vel_pub.publish(Twist()) 
                return True, dist
            else:
                raw_z = error_x * self.P_GAIN
                if raw_z > 0:
                    ang_z = min(raw_z, self.MAX_ROT_SPEED)
                else:
                    ang_z = max(raw_z, -self.MAX_ROT_SPEED)
                cmd = Twist()
                cmd.angular.z = float(ang_z)
                self.cmd_vel_pub.publish(cmd)
                return False, dist
        
        cmd = Twist()
        cmd.angular.z = 0.2
        self.cmd_vel_pub.publish(cmd)
        cv2.imshow("Centering Nurse", self.cv_image)
        cv2.waitKey(1)
        return False, 99.9

    def calculate_global_coords(self, distance):
        rx, ry, ryaw = self.rx, self.ry, self.ryaw
        nx = rx + distance * math.cos(ryaw)
        ny = ry + distance * math.sin(ryaw)
        self.nurse_global_x = nx
        self.nurse_global_y = ny
        self.get_logger().info(f"📍 Nurse Locked: ({nx:.2f}, {ny:.2f})")

    # --- [MODIFIED] Use Hardcoded Yaw Values ---
    def generate_waypoints(self):
        nx, ny = self.nurse_global_x, self.nurse_global_y
        offset = math.sqrt(2) / 2.0 
        
        # Tuple Format: (Offset X, Offset Y, Hardcoded Yaw)
        # Based on your notes: 
        # 1: (x+off, y+off) -> 3.14
        # 2: (x-off, y+off) -> 3.14
        # 3: (x-off, y-off) -> -1.57
        # 4: (x+off, y-off) -> 0.0
        # 5: (x+off, y+off) -> 1.57
        
        waypoints_def = [
            (offset,  offset,  3.14),   # Point 1
            (-offset, offset,  3.14),   # Point 2
            (-offset, -offset, -1.57),  # Point 3
            (offset,  -offset, 0.0),    # Point 4
            (offset,  offset,  1.57)    # Point 5 (Return)
        ]
        
        self.waypoints = []
        for ox, oy, fixed_yaw in waypoints_def:
            wx = nx + ox
            wy = ny + oy
            self.waypoints.append({'x': wx, 'y': wy, 'yaw': fixed_yaw})
            
        self.get_logger().info(f"🗺️ Generated {len(self.waypoints)} waypoints with fixed yaw.")

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

    def mission_loop(self):
        if self.robot_pose is None: return
        if self.start_delay < 10: 
            self.start_delay += 1
            return
        
        if self.is_sleeping(): return

        if self.state == 0:
            self.send_nav_command(self.ROOM_ENTRANCE)
            self.state = 1 

        elif self.state == 1: pass 

        elif self.state == 2:
            is_centered, dist = self.process_centering()
            
            if is_centered:
                if dist < 5.0:
                    self.get_logger().info(f"✅ Target Locked! Distance: {dist:.2f}m")
                    self.set_sleep(1.0) 
                    self.calculate_global_coords(dist)
                    self.state = 3
                else:
                    self.get_logger().warn("⚠️ Centered but depth invalid.")

        elif self.state == 3:
            self.generate_waypoints()
            self.state = 4
            self.current_waypoint_idx = 0

        elif self.state == 4:
            if not self.is_navigating:
                if self.current_waypoint_idx < len(self.waypoints):
                    wp = self.waypoints[self.current_waypoint_idx]
                    self.get_logger().info(f"🚶 Waypoint {self.current_waypoint_idx+1}")
                    self.send_nav_command(wp)
                    self.current_waypoint_idx += 1
                else:
                    self.get_logger().info("🎉 Mission Complete!")
                    self.speech_pub.publish(String(data="bark"))
                    self.state = 5

def main(args=None):
    rclpy.init(args=args)
    node = Mission6NurseController()
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