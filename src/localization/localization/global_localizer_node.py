#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, TransformStamped, PoseArray, Pose
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan, Imu
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import tf_transformations
import numpy as np
from collections import deque

# ----------------- 유틸리티 -----------------
def pose_to_matrix(x, y, yaw):
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    c = np.cos(yaw)
    s = np.sin(yaw)
    T[0, 0] = c; T[0, 1] = -s
    T[1, 0] = s; T[1, 1] = c
    return T

class LocalizationNode(Node):
    def __init__(self):
        super().__init__('localization_node')
        self.get_logger().info('High Precision Particle Filter Started!')

        # [1] 파라미터 선언
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 1.0)
        self.declare_parameter('z', 0.5)
        self.declare_parameter('roll', 0.0)
        self.declare_parameter('pitch', 0.0)
        self.declare_parameter('yaw', 0.0)
        self.declare_parameter('num_particles', 2000) # 파티클 수는 유지
        
        self.init_x = self.get_parameter('x').value
        self.init_y = self.get_parameter('y').value
        self.init_z = self.get_parameter('z').value
        self.init_roll = self.get_parameter('roll').value
        self.init_pitch = self.get_parameter('pitch').value
        self.init_yaw = self.get_parameter('yaw').value
        self.num_particles = self.get_parameter('num_particles').value

        # [2] 튜닝 파라미터 (정밀도 극대화 세팅)
        # 시뮬레이션 Odom은 정확하므로 노이즈를 아주 작게 잡음
        self.alpha1 = 0.1  # 회전 -> 회전 노이즈 (매우 낮춤)
        self.alpha2 = 0.02  # 이동 -> 회전 노이즈
        self.alpha3 = 0.05   # 이동 -> 이동 노이즈
        self.alpha4 = 0.1  # 회전 -> 이동 노이즈
        
        self.max_beams = 60    # 빔 개수 2배 증가 (정밀도 향상)
        self.sigma_hit = 0.3   # [핵심] 0.5 -> 0.15 (센서가 칼같이 맞아야 점수 줌)
        self.z_rand = 0.1       # 랜덤 노이즈 비중 축소 (센서 신뢰도 상승)
        self.resample_threshold = 0.5
        self.random_injection_ratio = 0.02 # 랜덤 주입 1%로 축소 (안정성 중시)

        # [3] 내부 변수
        self.particles = None 
        self.weights = None    
        self.map_grid = None   
        self.map_info = None   
        self.distance_field = None
        self.last_odom = None  
        self.current_map_odom = None

        # 라이다 오프셋
        self.laser_offset_x = 0.0
        self.laser_offset_y = 0.0
        self.laser_offset_yaw = 0.0
        self.offset_initialized = False

        self.imu_yaw = None
        self.imu_yaw_var = 0.05 ** 2
        self.last_imu_time = None
        self.imu_timeout = 0.5  # seconds

        # [4] ROS 통신
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu_plugin/out', self.imu_callback, 50)
        
        self.pose_pub = self.create_publisher(PoseStamped, '/go1_pose', 10)
        self.cloud_pub = self.create_publisher(PoseArray, '/particle_cloud', 10)

    # ----------------- 1. 지도 초기화 -----------------
    def map_callback(self, msg):
        self.map_grid = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        self.distance_field = self.compute_distance_transform(self.map_grid, msg.info.resolution)
        self.get_logger().info('Map Received! Initializing Particles...')
        self.initialize_particles()

    def initialize_particles(self):
        self.particles = np.zeros((self.num_particles, 3))
        # 초기 분산도 줄임 (0.5 -> 0.2)
        self.particles[:, 0] = self.init_x + np.random.normal(0, 0.2, self.num_particles)
        self.particles[:, 1] = self.init_y + np.random.normal(0, 0.2, self.num_particles)
        self.particles[:, 2] = self.init_yaw + np.random.normal(0, 0.1, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def imu_callback(self, msg: Imu):
        q = msg.orientation
        yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.imu_yaw = yaw
        cov = msg.orientation_covariance
        if len(cov) == 9 and cov[8] > 0:
            self.imu_yaw_var = max(cov[8], 1e-5)
        now = self.get_clock().now()
        self.last_imu_time = now.nanoseconds * 1e-9

    # ----------------- 2. 메인 루프 -----------------
    def scan_callback(self, scan_msg):
        if self.particles is None or self.map_grid is None: return

        try:
            trans_odom = self.tf_buffer.lookup_transform('odom', 'base', Time(seconds=0))
            if not self.offset_initialized:
                try:
                    trans_laser = self.tf_buffer.lookup_transform('base', scan_msg.header.frame_id, Time(seconds=0))
                    self.laser_offset_x = trans_laser.transform.translation.x
                    self.laser_offset_y = trans_laser.transform.translation.y
                    q_l = trans_laser.transform.rotation
                    self.laser_offset_yaw = tf_transformations.euler_from_quaternion([q_l.x, q_l.y, q_l.z, q_l.w])[2]
                    self.offset_initialized = True
                    self.get_logger().info(f"Offset Set: {self.laser_offset_x:.3f}, {self.laser_offset_yaw:.3f}")
                except:
                    pass
        except:
            return

        q = trans_odom.transform.rotation
        yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        current_odom = np.array([trans_odom.transform.translation.x, trans_odom.transform.translation.y, yaw])

        if self.last_odom is not None:
            dx = current_odom[0] - self.last_odom[0]
            dy = current_odom[1] - self.last_odom[1]
            dyaw = current_odom[2] - self.last_odom[2]

            delta_trans = np.hypot(dx, dy)
            heading = np.arctan2(dy, dx)
            delta_rot1 = self.normalize_angle(heading - self.last_odom[2]) if delta_trans > 1e-6 else 0.0
            delta_rot2 = self.normalize_angle(dyaw - delta_rot1)
            
            # 아주 미세한 움직임도 반영 (0.001 -> 0.0001)
            if delta_trans > 0.0001 or np.abs(dyaw) > 0.0001:
                self.motion_update(delta_trans, delta_rot1, delta_rot2)
                self.sensor_update_vectorized(scan_msg)
                self.resample_particles()
        
        self.last_odom = current_odom
        self.publish_results(scan_msg.header.stamp)

    # ----------------- 3. Motion Update -----------------
    def motion_update(self, delta_trans, delta_rot1, delta_rot2):
        sigma_rot1 = self.alpha1 * abs(delta_rot1) + self.alpha2 * delta_trans
        sigma_trans = self.alpha3 * delta_trans + self.alpha4 * (abs(delta_rot1) + abs(delta_rot2))
        sigma_rot2 = self.alpha1 * abs(delta_rot2) + self.alpha2 * delta_trans

        delta_rot1_hat = delta_rot1 + np.random.normal(0, sigma_rot1, self.num_particles)
        delta_trans_hat = delta_trans + np.random.normal(0, sigma_trans, self.num_particles)
        delta_rot2_hat = delta_rot2 + np.random.normal(0, sigma_rot2, self.num_particles)

        theta = self.particles[:, 2] + delta_rot1_hat
        self.particles[:, 0] += delta_trans_hat * np.cos(theta)
        self.particles[:, 1] += delta_trans_hat * np.sin(theta)
        self.particles[:, 2] = self.normalize_angle(theta + delta_rot2_hat)

    # ----------------- 4. Sensor Update -----------------
    def sensor_update_vectorized(self, scan_msg):
        if self.distance_field is None: return

        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        # Max Range 필터링 (튀는 현상 방지)
        valid_mask = (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max - 0.1)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0: return

        step = max(1, len(valid_indices) // self.max_beams)
        sampled_indices = valid_indices[::step]
        sampled_ranges = ranges[sampled_indices]
        laser_angles = scan_msg.angle_min + sampled_indices * scan_msg.angle_increment

        p_x = self.particles[:, 0].reshape(-1, 1)
        p_y = self.particles[:, 1].reshape(-1, 1)
        p_yaw = self.particles[:, 2].reshape(-1, 1)

        sensor_x = p_x + (self.laser_offset_x * np.cos(p_yaw) - self.laser_offset_y * np.sin(p_yaw))
        sensor_y = p_y + (self.laser_offset_x * np.sin(p_yaw) + self.laser_offset_y * np.cos(p_yaw))
        sensor_yaw = p_yaw + self.laser_offset_yaw

        global_angles = sensor_yaw + laser_angles.reshape(1, -1)
        hit_x = sensor_x + sampled_ranges.reshape(1, -1) * np.cos(global_angles)
        hit_y = sensor_y + sampled_ranges.reshape(1, -1) * np.sin(global_angles)

        map_res = self.map_info.resolution
        map_ox = self.map_info.origin.position.x
        map_oy = self.map_info.origin.position.y
        map_w = self.map_info.width
        map_h = self.map_info.height

        map_idx_x = ((hit_x - map_ox) / map_res).astype(int)
        map_idx_y = ((hit_y - map_oy) / map_res).astype(int)

        in_map_mask = (map_idx_x >= 0) & (map_idx_x < map_w) & \
                      (map_idx_y >= 0) & (map_idx_y < map_h)
        
        safe_x = np.clip(map_idx_x, 0, map_w - 1)
        safe_y = np.clip(map_idx_y, 0, map_h - 1)

        dist_errors = self.distance_field[safe_y, safe_x]

        sigma_sq = self.sigma_hit ** 2
        hit_prob = np.exp(-0.5 * (dist_errors ** 2) / sigma_sq)
        
        final_prob = np.where(in_map_mask, hit_prob, self.z_rand)
        final_prob = final_prob * (1.0 - self.z_rand) + self.z_rand

        log_weights = np.sum(np.log(final_prob + 1e-10), axis=1)

        if self.imu_yaw is not None and self.last_imu_time is not None:
            age = self.get_clock().now().nanoseconds * 1e-9 - self.last_imu_time
            if age < self.imu_timeout:
                yaw_diff = self.normalize_angle(self.particles[:, 2] - self.imu_yaw)
                imu_sigma = max(self.imu_yaw_var, 1e-5)
                log_weights += -0.5 * (yaw_diff ** 2) / imu_sigma
        log_weights -= np.max(log_weights)
        self.weights = np.exp(log_weights)
        
        w_sum = np.sum(self.weights)
        if w_sum > 0:
            self.weights /= w_sum
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

    # ----------------- 5. Resampling -----------------
    def resample_particles(self):
        n_eff = 1.0 / np.sum(self.weights ** 2)
        threshold = self.resample_threshold * self.num_particles

        if n_eff > threshold: return

        new_particles = np.zeros_like(self.particles)
        r = np.random.uniform(0, 1.0 / self.num_particles)
        c = self.weights[0]
        i = 0
        
        for m in range(self.num_particles):
            u = r + m * (1.0 / self.num_particles)
            while u > c:
                i = (i + 1) % self.num_particles
                c += self.weights[i]
            new_particles[m] = self.particles[i]

        num_random = int(self.num_particles * self.random_injection_ratio)
        if num_random > 0:
            random_indices = np.random.choice(self.num_particles, num_random, replace=False)
            mean_x = np.mean(new_particles[:, 0])
            mean_y = np.mean(new_particles[:, 1])
            new_particles[random_indices, 0] = mean_x + np.random.normal(0, 1.0, num_random)
            new_particles[random_indices, 1] = mean_y + np.random.normal(0, 1.0, num_random)
            new_particles[random_indices, 2] = np.random.uniform(-np.pi, np.pi, num_random)

        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles

    def publish_results(self, timestamp):
        # [핵심 수정] 1등만 믿지 말고, 상위 20% 우등생들의 평균을 사용
        # 이유: 1등은 노이즈에 민감하고, 전체 평균은 부정확함. 그 중간이 최고.
        
        # 1. 가중치 기준으로 정렬 (오름차순)
        sorted_indices = np.argsort(self.weights)
        
        # 2. 상위 20%만 남기기
        top_k = max(1, int(self.num_particles * 0.2)) # 상위 20%
        top_indices = sorted_indices[-top_k:] # 뒤에서부터 k개 (가중치 높은 순)
        
        top_particles = self.particles[top_indices]
        top_weights = self.weights[top_indices]
        
        # 3. 상위 그룹 내에서 다시 가중치 정규화
        w_sum = np.sum(top_weights)
        if w_sum > 0:
            top_weights /= w_sum
        else:
            top_weights = np.ones(top_k) / top_k

        # 4. 상위 그룹의 가중 평균 계산
        mean_x = np.average(top_particles[:, 0], weights=top_weights)
        mean_y = np.average(top_particles[:, 1], weights=top_weights)
        sin_mean = np.average(np.sin(top_particles[:, 2]), weights=top_weights)
        cos_mean = np.average(np.cos(top_particles[:, 2]), weights=top_weights)
        mean_yaw = np.arctan2(sin_mean, cos_mean)

        # -----------------------------------------------------------
        
        # 1. Pose Pub
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = mean_x
        pose_msg.pose.position.y = mean_y
        pose_msg.pose.position.z = self.init_z 
        q = tf_transformations.quaternion_from_euler(self.init_roll, self.init_pitch, mean_yaw)
        pose_msg.pose.orientation.x = q[0]; pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]; pose_msg.pose.orientation.w = q[3]
        self.pose_pub.publish(pose_msg)

        # 2. TF Pub (Smoothing 적용)
        self.broadcast_tf(mean_x, mean_y, mean_yaw, timestamp)

        # 3. Cloud Pub (상위 20%만 빨간색으로 시각화하면 더 보기 좋음 - 여기선 전체 표시)
        self.publish_cloud(pose_msg.header)

    def broadcast_tf(self, x, y, yaw, timestamp):
        try:
            T_map_base = pose_to_matrix(x, y, yaw)
            trans = self.tf_buffer.lookup_transform('odom', 'base', Time(seconds=0))
            t = trans.transform.translation
            r = trans.transform.rotation
            yaw_odom = tf_transformations.euler_from_quaternion([r.x, r.y, r.z, r.w])[2]
            T_odom_base = pose_to_matrix(t.x, t.y, yaw_odom)
            T_target = np.dot(T_map_base, np.linalg.inv(T_odom_base))
            
            # 1등 파티클은 위치가 틱틱 바뀔 수 있으므로 Smoothing을 조금 더 강하게 줌
            alpha = 0.3 # 부드럽게 따라가도록
            if self.current_map_odom is None:
                self.current_map_odom = T_target
            else:
                self.current_map_odom[:3, 3] = (1 - alpha) * self.current_map_odom[:3, 3] + alpha * T_target[:3, 3]
                curr_yaw = tf_transformations.euler_from_matrix(self.current_map_odom)[2]
                target_yaw = tf_transformations.euler_from_matrix(T_target)[2]
                diff = self.normalize_angle(target_yaw - curr_yaw)
                new_yaw = curr_yaw + alpha * diff
                self.current_map_odom[:3, :3] = tf_transformations.euler_matrix(0, 0, new_yaw)[:3, :3]

            tx = self.current_map_odom[0, 3]
            ty = self.current_map_odom[1, 3]
            tyaw = tf_transformations.euler_from_matrix(self.current_map_odom)[2]
            
            t_msg = TransformStamped()
            t_msg.header.stamp = timestamp
            t_msg.header.frame_id = "map"
            t_msg.child_frame_id = "odom"
            t_msg.transform.translation.x = tx
            t_msg.transform.translation.y = ty
            t_msg.transform.translation.z = 0.0 
            q_final = tf_transformations.quaternion_from_euler(0, 0, tyaw)
            t_msg.transform.rotation.x = q_final[0]; t_msg.transform.rotation.y = q_final[1]
            t_msg.transform.rotation.z = q_final[2]; t_msg.transform.rotation.w = q_final[3]
            
            self.tf_broadcaster.sendTransform(t_msg)
        except Exception:
            pass

    def publish_cloud(self, header):
        cloud_msg = PoseArray()
        cloud_msg.header = header
        for p in self.particles[::10]:
            pt = Pose()
            pt.position.x = p[0]
            pt.position.y = p[1]
            pt.position.z = self.init_z
            q = tf_transformations.quaternion_from_euler(0, 0, p[2])
            pt.orientation.x = q[0]; pt.orientation.y = q[1]
            pt.orientation.z = q[2]; pt.orientation.w = q[3]
            cloud_msg.poses.append(pt)
        self.cloud_pub.publish(cloud_msg)

    def compute_distance_transform(self, grid, resolution):
        h, w = grid.shape
        dist = np.full((h, w), np.inf, dtype=np.float32)
        occupied = grid >= 50
        queue = deque()

        occ_indices = np.argwhere(occupied)
        if occ_indices.size == 0:
            dist.fill(max(h, w) * resolution)
            return dist

        for y, x in occ_indices:
            dist[y, x] = 0.0
            queue.append((y, x))

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            y, x = queue.popleft()
            base = dist[y, x]
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    alt = base + 1.0
                    if alt < dist[ny, nx]:
                        dist[ny, nx] = alt
                        queue.append((ny, nx))

        dist *= resolution
        return dist

    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

def main(args=None):
    rclpy.init(args=args)
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
