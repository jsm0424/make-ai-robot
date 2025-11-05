#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from slam_toolbox.srv import SaveMap
from ament_index_python.packages import get_package_share_directory

PKG_NAME = 'go1_simulation'
MAP_BASENAME = 'my_world'  # change if you want a different filename

def compute_src_maps_path(pkg_name: str) -> str:
    """
    Compute <workspace_root>/src/<pkg_name>/maps from the installed share path,
    without hardcoding any absolute user-specific paths.
    """
    share_dir = get_package_share_directory(pkg_name)
    # Typical pattern: <ws>/install/<pkg>/share/<pkg>
    parts = share_dir.split(os.sep)
    try:
        install_idx = parts.index('install')  # find workspace/install
        ws_root = os.sep.join(parts[:install_idx])  # <ws>
    except ValueError:
        # Fallback: if not in an installed workspace, go up until we find 'src'
        # and assume workspace root is the parent of 'src'.
        cur = share_dir
        ws_root = None
        while True:
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            if os.path.basename(parent) == 'src':
                ws_root = os.path.dirname(parent)
                break
            cur = parent
        if ws_root is None:
            # Last resort: current working directory as workspace root
            ws_root = os.getcwd()
    return os.path.join(ws_root, 'src', pkg_name, 'maps')

class MapSaver(Node):
    def __init__(self):
        super().__init__('save_map')
        self.cli = self.create_client(SaveMap, '/slam_toolbox/save_map')

        # Resolve src/<pkg>/maps path in a workspace-agnostic way
        self.maps_dir = compute_src_maps_path(PKG_NAME)
        os.makedirs(self.maps_dir, exist_ok=True)
        self.target_base = os.path.join(self.maps_dir, MAP_BASENAME)

        self.get_logger().info(f"Saving map to: {self.target_base} (.pgm/.yaml)")
        self.get_logger().info('Waiting for /slam_toolbox/save_map service...')
        self.cli.wait_for_service()
        self.get_logger().info('Service available, sending request...')

        req = SaveMap.Request()
        # IMPORTANT: SaveMap expects a std_msgs/String for 'name'
        req.name.data = self.target_base

        self.future = self.cli.call_async(req)
        self.future.add_done_callback(self._on_done)

    def _on_done(self, future):
        try:
            resp = future.result()
            if resp.result == 0:
                self.get_logger().info(f"✅ Map saved: {self.target_base}.pgm/.yaml")
            else:
                self.get_logger().error(f"❌ Map save failed (result code: {resp.result})")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
        finally:
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = MapSaver()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
