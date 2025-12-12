#include <memory>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"

#include "astar_planner/astar.hpp"

using namespace std::chrono_literals;

class PathPlannerNode : public rclcpp::Node
{
public:
  PathPlannerNode()
  : Node("path_planner_node")
  {
    this->declare_parameter<double>("resolution", 1.0);
    resolution_ = this->get_parameter("resolution").as_double();
    
    // 상태 변수 초기화
    has_map_ = false;
    has_goal_ = false;
    has_current_pose_ = false;
    goal_reached_ = false;

    // QoS 설정 (Map은 Reliable & Transient Local 필수)
    rclcpp::QoS map_qos_profile(10);
    map_qos_profile.transient_local(); 
    map_qos_profile.reliable();  
    
    // Subscribers
    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/map", map_qos_profile,
      std::bind(&PathPlannerNode::mapCallback, this, std::placeholders::_1));
    
    current_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/go1_pose", 10,
      std::bind(&PathPlannerNode::currentPoseCallback, this, std::placeholders::_1));
    
    goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/goal_pose", 10,
      std::bind(&PathPlannerNode::goalCallback, this, std::placeholders::_1));

    // Publishers
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/local_path", 10);
    viz_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/path_markers", 10);
    goal_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/goal_marker", 10);
    
    RCLCPP_INFO(this->get_logger(), "Path Planner: Static Map Mode Initialized");
  }

private:
  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    map_msg_ = msg;
    int width = msg->info.width;
    int height = msg->info.height;
    
    // 맵 데이터를 2차원 벡터로 변환 (0: Free, 1: Obstacle)
    map_grid_.clear();
    map_grid_.resize(height, std::vector<int>(width));
    
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int index = y * width + x;
        // OccupancyGrid 값: -1(Unknown), 0(Free), 100(Occupied)
        if (msg->data[index] > 50 || msg->data[index] < 0) {
          map_grid_[y][x] = 1; // 장애물
        } else {
          map_grid_[y][x] = 0; // 이동 가능
        }
      }
    }
    
    // A* 라이브러리에 지도 전달
    astar_.setMap(map_grid_);

    if (!has_map_) {
      has_map_ = true;
      RCLCPP_INFO(this->get_logger(), "Map received: %dx%d", width, height);
    }
  }
  
  void currentPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (!has_current_pose_) {
      has_current_pose_ = true;
      previous_pose_ = *msg;
    }
    current_pose_ = *msg;

    // 목표 지점에 도착했는지 확인 (거리 0.5m 이내)
    if (has_goal_) {
      double goal_dx = current_pose_.pose.position.x - goal_pose_.pose.position.x;
      double goal_dy = current_pose_.pose.position.y - goal_pose_.pose.position.y;
      double dist = std::sqrt(goal_dx * goal_dx + goal_dy * goal_dy);
      
      if (dist < 0.5 && !goal_reached_) {
          RCLCPP_INFO(this->get_logger(), "✓ Goal reached!");
          goal_reached_ = true;
      }
    }
    
    // Static Mode에서는 로봇이 움직인다고 경로를 재계산하지 않음 (replanPath 호출 X)
    // 필요하다면 여기서 일정 거리 이상 움직였을 때 replanPath()를 호출할 수 있음.
  }
  
  void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    goal_pose_ = *msg;
    has_goal_ = true;
    goal_reached_ = false;
    
    RCLCPP_INFO(this->get_logger(), "New goal received -> Planning Path...");
    publishGoalMarker();
    
    // 목표가 들어오면 정적 맵 기반으로 경로 계산
    if (has_map_ && has_current_pose_) {
      replanPath(); 
    }
  }

  geometry_msgs::msg::Quaternion createQuaternionMsgFromYaw(double yaw)
  {
    geometry_msgs::msg::Quaternion q;
    q.x = 0.0; q.y = 0.0;
    q.z = std::sin(yaw / 2.0); q.w = std::cos(yaw / 2.0);
    return q;
  }

  void replanPath()
  {
    if (!has_map_ || !has_current_pose_ || !has_goal_) return;
    
    // [중요] Scan 데이터 없이 기존 map_grid_를 그대로 사용
    // mapCallback에서 이미 astar_.setMap(map_grid_)를 했지만 안전을 위해 확인
    astar_.setMap(map_grid_);
    
    astar_planner::GridCell start = worldToGrid(current_pose_.pose.position.x, current_pose_.pose.position.y);
    astar_planner::GridCell goal = worldToGrid(goal_pose_.pose.position.x, goal_pose_.pose.position.y);
    
    // A* 실행
    auto path_cells = astar_.findPath(start, goal);
    
    if (path_cells.empty()) {
      RCLCPP_WARN(this->get_logger(), "No path found!");
      return;
    }
    
    // Path 메시지 생성
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->now();
    path_msg.header.frame_id = "map";
    
    for (size_t i = 0; i < path_cells.size(); ++i) {
      const auto& cell = path_cells[i];
      auto world_pos = gridToWorld(cell.x, cell.y);
      geometry_msgs::msg::PoseStamped pose;
      pose.header.stamp = this->now();
      pose.header.frame_id = "map";
      pose.pose.position.x = world_pos.first;
      pose.pose.position.y = world_pos.second;
      pose.pose.position.z = 0.0;
      
      // 경로의 방향(Orientation) 설정: 다음 점을 바라보도록 함
      if(i < path_cells.size() - 1){
        const auto& next_cell = path_cells[i+1];
        auto next_world_pos = gridToWorld(next_cell.x, next_cell.y);
        double dx = next_world_pos.first - world_pos.first;
        double dy = next_world_pos.second - world_pos.second;
        double yaw = std::atan2(dy, dx);
        pose.pose.orientation = createQuaternionMsgFromYaw(yaw);
      } else {
        // 마지막 점은 목표 지점의 방향을 따름
        pose.pose.orientation = goal_pose_.pose.orientation;
      }
      path_msg.poses.push_back(pose);
    }
    
    path_pub_->publish(path_msg);
    publishPathMarkers(path_cells);
    RCLCPP_INFO(this->get_logger(), "Path planned successfully (%zu points)", path_cells.size());
  }
  
  astar_planner::GridCell worldToGrid(double x, double y)
  {
    astar_planner::GridCell cell;
    double origin_x = map_msg_->info.origin.position.x;
    double origin_y = map_msg_->info.origin.position.y;
    double resolution = map_msg_->info.resolution;
    cell.x = static_cast<int>((x - origin_x) / resolution);
    cell.y = static_cast<int>((y - origin_y) / resolution);
    return cell;
  }
  
  std::pair<double, double> gridToWorld(int x, int y)
  {
    double origin_x = map_msg_->info.origin.position.x;
    double origin_y = map_msg_->info.origin.position.y;
    double resolution = map_msg_->info.resolution;
    double world_x = origin_x + (x + 0.5) * resolution;
    double world_y = origin_y + (y + 0.5) * resolution;
    return {world_x, world_y};
  }
  
  void publishPathMarkers(const std::vector<astar_planner::GridCell>& path)
  {
    visualization_msgs::msg::MarkerArray marker_array;
    visualization_msgs::msg::Marker line_marker;
    line_marker.header.frame_id = "map";
    line_marker.header.stamp = this->now();
    line_marker.ns = "path";
    line_marker.id = 0;
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::msg::Marker::ADD;
    line_marker.scale.x = 0.1;
    line_marker.color.r = 0.0; line_marker.color.g = 1.0; line_marker.color.b = 0.0; line_marker.color.a = 1.0;
    
    for (const auto& cell : path) {
      geometry_msgs::msg::Point p;
      auto world_pos = gridToWorld(cell.x, cell.y);
      p.x = world_pos.first; p.y = world_pos.second; p.z = 0.1;
      line_marker.points.push_back(p);
    }
    marker_array.markers.push_back(line_marker);
    viz_pub_->publish(marker_array);
  }
  
  void publishGoalMarker()
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->now();
    marker.ns = "goal";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose = goal_pose_.pose;
    marker.scale.x = 0.8; marker.scale.y = 0.8; marker.scale.z = 0.8;
    marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0; marker.color.a = 0.8;
    goal_marker_pub_->publish(marker);
  }
  
  // ROS objects
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr current_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_marker_pub_;
  
  // State variables
  bool has_map_;
  bool has_goal_;
  bool has_current_pose_;
  bool goal_reached_;
  
  nav_msgs::msg::OccupancyGrid::SharedPtr map_msg_;
  
  geometry_msgs::msg::PoseStamped current_pose_;
  geometry_msgs::msg::PoseStamped previous_pose_;
  geometry_msgs::msg::PoseStamped goal_pose_;
  
  std::vector<std::vector<int>> map_grid_;
  astar_planner::AStar astar_;
  
  double resolution_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathPlannerNode>());
  rclcpp::shutdown();
  return 0;
}