#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
from datetime import datetime

class ImageSaver(Node):
    """
    카메라 토픽을 구독하여 1초 간격으로 이미지 파일(.jpg)로 저장하는 노드입니다.
    """

    def __init__(self):
        super().__init__('image_saver')

        # 파라미터 선언
        self.declare_parameter('image_topic', '/camera_top/image') 
        self.declare_parameter('save_folder', 'test_images')
        
        # 파라미터 가져오기
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        folder_name = self.get_parameter('save_folder').get_parameter_value().string_value

        # CV Bridge 초기화
        self.bridge = CvBridge()
        
        # 저장 경로 생성
        self.save_dir = os.path.join(os.getcwd(), folder_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            self.get_logger().info(f'폴더 생성됨: {self.save_dir}')

        # 카메라 구독
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        self.get_logger().info(f'Image Saver 시작됨 (1초 간격 녹화).')
        self.get_logger().info(f'구독 중인 토픽: {self.image_topic}')
        self.get_logger().info(f'저장 경로: {self.save_dir}')
        
        # 상태 변수
        self.resolution_checked = False
        self.last_save_time = 0.0  # 마지막 저장 시간 초기화

    def image_callback(self, msg):
        try:
            # 현재 시간 확인
            current_time = time.time()
            
            # 마지막 저장 후 1초가 지나지 않았으면 무시 (Throttling)
            if current_time - self.last_save_time < 1.0:
                return

            # 1. ROS 이미지를 OpenCV 포맷으로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 2. 해상도 확인 (최초 1회)
            if not self.resolution_checked:
                height, width, _ = cv_image.shape
                self.get_logger().info(f'수신된 이미지 해상도: {width}x{height} (원본 유지)')
                self.resolution_checked = True

            # 3. 저장 로직 실행
            self.last_save_time = current_time
            
            # 파일명 생성 (밀리초 제거, 초 단위까지만 기록)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"img_{timestamp}.jpg"
            filepath = os.path.join(self.save_dir, filename)
            
            # 4. 이미지 저장
            success = cv2.imwrite(filepath, cv_image)
            
            if success:
                self.get_logger().info(f'저장됨: {filename}')
            else:
                self.get_logger().warn(f'저장 실패: {filename}')

        except Exception as e:
            self.get_logger().error(f'이미지 처리 실패: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('사용자에 의해 중지됨.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()