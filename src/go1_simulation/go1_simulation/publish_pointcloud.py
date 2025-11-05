#! /usr/bin/env python3

"""
This script publishes pointclouds from multiple depth cameras.

The node subscribes to synchronized depth and RGB images from both 'face' and 'top' cameras,
converts them to 3D points in their respective camera frames.

Input topics (for each camera):
    - camera_face/camera_info: Face camera intrinsic parameters (focal length, principal point)
    - camera_face/image: RGB image from the face camera
    - camera_face/depth: Depth image from the face camera
    - camera_top/camera_info: Top camera intrinsic parameters (focal length, principal point)
    - camera_top/image: RGB image from the top camera
    - camera_top/depth: Depth image from the top camera

Output topics:
    - camera_face/points: PointCloud2 message with XYZRGB points in the face camera frame
    - camera_top/points: PointCloud2 message with XYZRGB points in the top camera frame
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import numpy as np
import struct
from rclpy.time import Time
import time


class CameraHandler:
    """Handles point cloud generation for a single camera"""
    
    def __init__(self, node, camera_name):
        """
        Initialize camera handler for a specific camera.
        
        Args:
            node: Parent ROS2 node
            camera_name: Name of the camera (e.g., 'face', 'top')
        """
        self.node = node
        self.camera_name = camera_name
        self.camera_frame = f'camera_optical_{camera_name}'
        self.bridge = CvBridge()
        self.camera_info = None
        
        # Subscribe to camera info
        self.camera_info_sub = self.node.create_subscription(
            CameraInfo,
            f'camera_{camera_name}/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Synchronized subscribers for image and depth
        self.image_sub = Subscriber(self.node, Image, f'camera_{camera_name}/image')
        self.depth_sub = Subscriber(self.node, Image, f'camera_{camera_name}/depth')
        
        # Synchronize image and depth messages
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        
        # Publisher for point cloud
        self.pointcloud_pub = self.node.create_publisher(
            PointCloud2,
            f'camera_{camera_name}/points',
            10
        )
        
        self.node.get_logger().info(
            f'Camera handler initialized for {camera_name} camera '
            f'(frame: {self.camera_frame})'
        )
    
    def camera_info_callback(self, msg):
        """Store camera intrinsics"""
        if self.camera_info is None:
            self.camera_info = msg
            self.node.get_logger().info(f'Camera info received for {self.camera_name}')
    
    def sync_callback(self, image_msg, depth_msg):
        """Process synchronized image and depth messages"""
        if self.camera_info is None:
            self.node.get_logger().warn(
                f'Waiting for {self.camera_name} camera info...', 
                throttle_duration_sec=1.0
            )
            return
        
        try:
            # Convert depth image to numpy array
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # Convert RGB image to numpy array
            rgb_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
            
            # Create point cloud
            pointcloud_msg = self.create_pointcloud(depth_image, rgb_image, depth_msg.header)
            
            # Publish point cloud
            self.pointcloud_pub.publish(pointcloud_msg)
            
        except Exception as e:
            self.node.get_logger().error(
                f'Error processing images for {self.camera_name}: {str(e)}'
            )
    
    def create_pointcloud(self, depth_image, rgb_image, header):
        """
        Create PointCloud2 message from depth and RGB images.
        
        Converts 2D depth and RGB images into 3D colored points using camera intrinsics,
        
        Args:
            depth_image: Depth image as numpy array (meters)
            rgb_image: RGB image as numpy array
            header: Original message header (timestamp will be preserved)
            
        Returns:
            PointCloud2 message with XYZRGB points in target frame
        """
        
        # Get camera intrinsics
        fx = self.camera_info.k[0]  # focal length x
        fy = self.camera_info.k[4]  # focal length y
        cx = self.camera_info.k[2]  # principal point x
        cy = self.camera_info.k[5]  # principal point y
        
        height, width = depth_image.shape
        
        # Convert depth to meters if needed (assuming depth is in meters already)
        x = depth_image.astype(np.float32)
        
        # Filter out invalid depth values (zero, negative, inf, nan)
        valid_mask = (x > 0) & (x < np.inf) & np.isfinite(x)
        
        # Check if there are any valid points
        if not np.any(valid_mask):
            self.node.get_logger().warn(
                f'No valid depth values in {self.camera_name} image', 
                throttle_duration_sec=1.0
            )
            # Return empty point cloud
            msg = PointCloud2()
            msg.header = header
            msg.header.frame_id = self.camera_frame
            msg.height = 1
            msg.width = 0
            msg.is_dense = False
            msg.is_bigendian = False
            msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            msg.point_step = 16
            msg.row_step = 0
            msg.data = bytes()
            return msg
        
        # Create coordinate arrays
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)
        
        # Extract only valid depth values and corresponding coordinates
        x_valid = x[valid_mask]
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        
        # Compute 3D coordinates in camera frame (only for valid points)
        x = x_valid
        y = - (u_valid - cx) * x_valid / fx
        z = - (v_valid - cy) * x_valid / fy     

        # Get RGB values for valid points
        r = rgb_image[:, :, 0][valid_mask].flatten()
        g = rgb_image[:, :, 1][valid_mask].flatten()
        b = rgb_image[:, :, 2][valid_mask].flatten()
        
        # Pack RGB into single float
        rgb = np.zeros(len(r), dtype=np.float32)
        for i in range(len(r)):
            rgb_int = (int(r[i]) << 16) | (int(g[i]) << 8) | int(b[i])
            rgb[i] = struct.unpack('f', struct.pack('i', rgb_int))[0]
        
        # Create point cloud data
        points = np.zeros(len(x), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.float32)
        ])
        
        points['x'] = x
        points['y'] = y
        points['z'] = z
        points['rgb'] = rgb
        
        # Create PointCloud2 message
        msg = PointCloud2()
        msg.header = header
        msg.header.frame_id = self.camera_frame
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = False
        msg.is_bigendian = False
        
        # Define fields
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.data = points.tobytes()
        
        return msg


class PointCloudPublisher(Node):
    """Main node that manages point cloud publishers for multiple cameras"""
    
    def __init__(self):
        super().__init__('pointcloud_publisher')
        
        # Create handlers for both face and top cameras
        self.face_handler = CameraHandler(self, 'face')
        self.top_handler = CameraHandler(self, 'top')
        
        self.get_logger().info('PointCloud publisher node initialized for face and top cameras')


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()