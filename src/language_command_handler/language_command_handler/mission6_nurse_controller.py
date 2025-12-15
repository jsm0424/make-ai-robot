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
        # Location 1: Room Entrance
        self.LOC_1_DOOR = {'x': -6.3788, 'y': -24.8077, 'yaw': -3.13}
        
        # Location 2: Inside Room (Fallback) - Extracted from your image
        self.LOC_2_INSIDE = {'x': -7.65, 'y': -25.25, 'yaw': -2.06}
        
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
        
        # State Machine:
        # 0: Move to Door (Loc 1)
        # 1: Wait for Arrival at Door
        # 2: Search/Center Nurse (First Attempt)
        # 2.5: Move to Inside (Loc 2) - Triggered if search fails
        # 2.6: Wait for Arrival Inside
        # 2.7: Search/Center Nurse (Second Attempt)
        # 3: Calculate Coords
        # 4: Generate Waypoints
        # 5: Execute Waypoints
        self.state = 0
        
        self.start_delay = 0 
        self.wait_until_time = 0.0
        self.is_navigating = False
        
        # Search Timer
        self.search_start_time = 0.0
        self.SEARCH_TIMEOUT = 60.0 # Give up location 1 after 15 seconds

        self.create_timer(0.1, self.mission_loop)
        self.get_logger().info("🧠 Mission 6: Dual-Location Search Strategy Started")

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
            
            # Arrived at Door (Loc 1)
            if self.state == 1: 
                self.get_logger().info("🚪 Arrived at Door. Starting Search 1.")
                self.set_sleep(2.0)
                self.search_start_time = self.get_clock().now().nanoseconds / 1e9
                self.state = 2
            
            # Arrived Inside (Loc 2)
            elif self.state == 2.6:
                self.get_logger().info("🏠 Arrived Inside. Starting Search 2.")
                self.set_sleep(2.0)
                self.search_start_time = self.get_clock().now().nanoseconds / 1e9
                self.state = 2.7 # Go to second search state

    def set_sleep(self, seconds):
        self.wait_until_time = self.get_clock().now().nanoseconds / 1e9 + seconds
        self.cmd_vel_pub.publish(Twist())

    def is_sleeping(self):
        return (self.get_clock().now().nanoseconds / 1e9) < self.wait_until_time

    def get_depth_dist(self, cx, cy):
        """Original single-pixel depth (kept for fallback/scanning)"""
        if self.latest_depth_image is None: return 99.9
        h, w = self.latest_depth_image.shape
        cx = int(np.clip(cx, 0, w-1))
        cy = int(np.clip(cy, 0, h-1))
        try:
            val = self.latest_depth_image[cy, cx]
            if np.isnan(val) or np.isinf(val) or val == 0: return 99.9
            return float(val)
        except: return 99.9

    # === NEW FUNCTION ADDED ===
    def get_navy_weighted_distance(self, x1, y1, x2, y2):
        """
        Calculates MEDIAN depth of 'navy' pixels within the bounding box.
        Prevents getting background depth (like between legs).
        """
        if self.cv_image is None or self.latest_depth_image is None:
            return None

        # 1. Define ROI (Region of Interest)
        h, w, _ = self.cv_image.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        roi_rgb = self.cv_image[y1:y2, x1:x2]
        roi_depth = self.latest_depth_image[y1:y2, x1:x2]

        if roi_rgb.size == 0 or roi_depth.size == 0:
            return None

        # 2. Create Navy/Blue Mask in HSV
        hsv_roi = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)
        
        # Navy Blue Range (Adjust these if needed)
        # H: 90-140 (Blueish), S: >50 (Not white), V: 20-255 (Not pitch black)
        lower_navy = np.array([90, 50, 20])
        upper_navy = np.array([140, 255, 255])
        
        mask = cv2.inRange(hsv_roi, lower_navy, upper_navy)

        # 3. Apply mask to depth
        # Select depth pixels where mask is > 0 (Navy pixels)
        target_depths = roi_depth[mask > 0]

        # 4. Filter invalid data (0, NaN, Inf)
        valid_depths = target_depths[~np.isnan(target_depths)]
        valid_depths = valid_depths[~np.isinf(valid_depths)]
        valid_depths = valid_depths[valid_depths > 0] 

        pixel_count = len(valid_depths)

        # Debug visualization of what the robot 'sees' as navy
        # cv2.imshow("Navy Mask", mask) 
        # cv2.waitKey(1)

        if pixel_count < 10:
            self.get_logger().warn(f"⚠️ Box found, but <10 Navy pixels detected. Lighting might be bad.")
            return None

        # 5. Calculate Median (Robust to outliers)
        median_dist = float(np.median(valid_depths))
        self.get_logger().info(f"🔵 Navy Scan: {pixel_count} px found | Median Dist: {median_dist:.3f}")
        
        return median_dist

    def process_centering(self):
        if self.cv_image is None or self.model is None: return False, 99.9

        results = self.model(self.cv_image, verbose=False, conf=0.5)
        _, img_w, _ = self.cv_image.shape
        center_x_screen = img_w / 2

        best_box = None
        max_conf = -1.0

        # Find best person
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0: 
                    if float(box.conf[0]) > max_conf:
                        max_conf = float(box.conf[0])
                        best_box = box
        
        # --- Logic: If Found ---
        if best_box:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Default to center pixel for now
            dist = self.get_depth_dist(cx, cy)
            error_x = center_x_screen - cx
            
            # Debug Draw
            cv2.rectangle(self.cv_image, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.line(self.cv_image, (int(center_x_screen), 0), (int(center_x_screen), 1000), (0,0,255), 1)
            # cv2.imshow("Centering Nurse", self.cv_image)
            # cv2.waitKey(1)

            # Check centered
            if abs(error_x) < self.CENTER_TOLERANCE:
                self.cmd_vel_pub.publish(Twist()) 
                
                # === MODIFIED LOGIC: Use Navy Pixel Median ===
                navy_dist = self.get_navy_weighted_distance(x1, y1, x2, y2)
                
                if navy_dist is not None:
                    final_dist = navy_dist
                    self.get_logger().info(f"🎯 Centered! Using Navy Median Dist: {final_dist:.3f}")
                else:
                    final_dist = dist # Fallback to center pixel
                    self.get_logger().warn(f"⚠️ Centered! Navy detection failed, using Center Pixel: {final_dist:.3f}")

                return True, final_dist
            else:
                # Proportional Control
                raw_z = error_x * self.P_GAIN
                if raw_z > 0: ang_z = min(raw_z, self.MAX_ROT_SPEED)
                else: ang_z = max(raw_z, -self.MAX_ROT_SPEED)
                
                cmd = Twist()
                cmd.angular.z = float(ang_z)
                cmd.linear.x = 0.1
                self.cmd_vel_pub.publish(cmd)
                return False, dist
        
        # --- Logic: Not Found (Scan) ---
        cmd = Twist()
        cmd.angular.z = 0.2 # Slow scan
        cmd.linear.x = 0.05
        self.cmd_vel_pub.publish(cmd)
        # cv2.imshow("Centering Nurse", self.cv_image)
        # cv2.waitKey(1)
        return False, 99.9

    def calculate_global_coords(self, distance):
        rx, ry, ryaw = self.rx, self.ry, self.ryaw
        nx = rx + distance * math.cos(ryaw)
        ny = ry + distance * math.sin(ryaw)
        self.nurse_global_x = nx-0.2
        self.nurse_global_y = ny-0.2
        self.get_logger().info(f"📍 Nurse Locked: ({self.nurse_global_x:.2f}, {self.nurse_global_y:.2f})")

    def generate_waypoints(self):
        nx, ny = self.nurse_global_x, self.nurse_global_y
        offset = math.sqrt(2) / 2.0 
        
        # (x_off, y_off, yaw)
        waypoints_def = [
            # (offset,  offset,  -3.13),
            (-offset, offset,  -3.13),
            (-offset, -offset, -1.57),
            (offset,  -offset, 0.0),
            (offset,  offset,  1.57)
        ]
        
        self.waypoints = []
        for ox, oy, fixed_yaw in waypoints_def:
            wx = nx + ox
            wy = ny + oy
            self.waypoints.append({'x': wx, 'y': wy, 'yaw': fixed_yaw})
            
        self.get_logger().info(f"🗺️ Generated {len(self.waypoints)} waypoints.")

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

        # [State 0] Move to Door
        if self.state == 0:
            self.send_nav_command(self.LOC_1_DOOR)
            self.state = 1 

        elif self.state == 1: pass 

        # [State 2] Search Attempt 1 (At Door)
        elif self.state == 2:
            is_centered, dist = self.process_centering()
            
            # Check for Timeout (Fail at Loc 1)
            elapsed = (self.get_clock().now().nanoseconds / 1e9) - self.search_start_time
            if elapsed > self.SEARCH_TIMEOUT:
                self.get_logger().warn("❌ Nurse not found at Door. Moving Inside!")
                self.cmd_vel_pub.publish(Twist()) # Stop spinning
                self.state = 2.5 # Transition to move inside
                return

            if is_centered:
                if dist < 8.0:
                    self.get_logger().info(f"✅ Found at Door! Dist: {dist:.2f}m")
                    self.set_sleep(1.0) 
                    self.calculate_global_coords(dist)
                    self.state = 3
                else:
                    self.get_logger().warn("⚠️ Centered but depth invalid.")

        # [State 2.5] Move Inside (Fallback)
        elif self.state == 2.5:
            self.send_nav_command(self.LOC_2_INSIDE)
            self.state = 2.6 # Wait for arrival

        elif self.state == 2.6: pass

        # [State 2.7] Search Attempt 2 (Inside)
        elif self.state == 2.7:
            # Same search logic, but no timeout needed (or can add another timeout if desired)
            is_centered, dist = self.process_centering()
            
            if is_centered:
                if dist < 8.0:
                    self.get_logger().info(f"✅ Found Inside! Dist: {dist:.2f}m")
                    self.set_sleep(1.0) 
                    self.calculate_global_coords(dist)
                    self.state = 3

        # [State 3] Generate Path
        elif self.state == 3:
            self.generate_waypoints()
            self.state = 4
            self.current_waypoint_idx = 0

        # [State 4] Execute Waypoints
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
