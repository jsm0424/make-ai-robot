#include "astar_planner/astar.hpp"
#include <iostream>
#include <limits>

namespace astar_planner
{

AStar::AStar()
: map_width_(0), map_height_(0)
{
}

AStar::~AStar()
{
}

void AStar::setMap(const std::vector<std::vector<int>>& map)
{
  map_ = map;
  if (!map_.empty()) {
    map_height_ = map_.size();
    map_width_ = map_[0].size();
  }
}

double AStar::calculateHeuristic(const GridCell& a, const GridCell& b) const
{
  // Euclidean distance or Manhattan distance
  double dx = static_cast<double>(a.x - b.x);
  double dy = static_cast<double>(a.y - b.y);
  return std::sqrt(dx * dx + dy * dy);
}

// [수정된 isValid 함수]
bool AStar::isValid(const GridCell& cell) const
{
  // 1. 맵 밖으로 나가는지 검사
  if (cell.x < 0 || cell.x >= map_width_ || cell.y < 0 || cell.y >= map_height_) {
    return false;
  }

  // 2. [추가됨] 안전 마진 (Safety Margin) 검사
  // 로봇의 덩치를 고려해서 장애물 주변 몇 칸도 장애물로 취급합니다.
  // 맵 해상도(resolution)가 0.05m라고 가정할 때:
  // margin = 3 이면 -> 0.15m (약 15cm) 여유를 둠
  // margin = 5 이면 -> 0.25m (약 25cm) 여유를 둠 (Go1 로봇에 추천)
  int margin = 3; 

  for (int dy = -margin; dy <= margin; ++dy) {
    for (int dx = -margin; dx <= margin; ++dx) {
      int nx = cell.x + dx;
      int ny = cell.y + dy;

      // 맵 범위 체크
      if (nx >= 0 && nx < map_width_ && ny >= 0 && ny < map_height_) {
        // 내 주변(margin 안쪽)에 장애물(1)이 하나라도 있으면
        // 여기는 로봇 몸통이 닿을 위험이 있으므로 '못 가는 길'로 판단!
        if (map_[ny][nx] == 1) {
          return false; 
        }
      }
    }
  }

  // 주변에 장애물이 하나도 없으면 통과
  return true;
}

std::vector<GridCell> AStar::getNeighbors(const GridCell& cell) const
{
  std::vector<GridCell> neighbors;
  
  // 8-connected grid: up, down, left, right, and 4 diagonals
  std::vector<std::pair<int, int>> directions = {
    {0, 1},   // up
    {0, -1},  // down
    {1, 0},   // right
    {-1, 0},  // left
    {1, 1},   // up-right
    {1, -1},  // down-right
    {-1, 1},  // up-left
    {-1, -1}  // down-left
  };
  
  for (const auto& dir : directions) {
    GridCell neighbor = {cell.x + dir.first, cell.y + dir.second};
    if (isValid(neighbor)) {
      neighbors.push_back(neighbor);
    }
  }
  
  return neighbors;
}

std::vector<GridCell> AStar::reconstructPath(
  const std::unordered_map<GridCell, GridCell, GridCellHash>& came_from,
  const GridCell& start,
  const GridCell& goal) const
{
  std::vector<GridCell> path;
  GridCell current = goal;
  
  while (!(current == start)) {
    path.push_back(current);
    auto it = came_from.find(current);
    if (it == came_from.end()) {
      break;
    }
    current = it->second;
  }
  
  path.push_back(start);
  std::reverse(path.begin(), path.end());
  
  return path;
}

// [추가] 주변에 벽이 있으면 벌점(Penalty)을 부과하는 함수
double AStar::calculateObstacleCost(const GridCell& cell)
{
  double penalty = 0.0;
  int search_radius = 12; // 검사할 범위 (Hard Margin보다 커야 함)

  for (int dy = -search_radius; dy <= search_radius; ++dy) {
    for (int dx = -search_radius; dx <= search_radius; ++dx) {
      int nx = cell.x + dx;
      int ny = cell.y + dy;

      // 맵 범위 체크
      if (nx >= 0 && nx < map_width_ && ny >= 0 && ny < map_height_) {
        // 만약 이 범위 안에 벽(1)이 있다면?
        if (map_[ny][nx] == 1) {
          // 거리 계산 (유클리드)
          double dist = std::sqrt(dx*dx + dy*dy);
          
          // 벽이랑 가까울수록 벌점을 세게 줌! (거리가 0이면 무한대니까 예외처리)
          if (dist > 0) {
            // 20.0은 가중치 계수입니다. 
            // 이 숫자가 클수록 로봇이 벽을 더 무서워해서 중앙으로 갑니다.
            penalty += (100.0 / dist); 
          }
        }
      }
    }
  }
  return penalty;
}

std::vector<GridCell> AStar::findPath(const GridCell& start, const GridCell& goal)
{
  std::vector<GridCell> empty_path;
  
  // Check if start and goal are valid
  if (!isValid(start)) {
    std::cerr << "Start position is invalid or occupied!" << std::endl;
    return empty_path;
  }
  
  if (!isValid(goal)) {
    std::cerr << "Goal position is invalid or occupied!" << std::endl;
    return empty_path;
  }
  
  // Priority queue for open set
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_set;
  
  // Track visited nodes
  std::unordered_map<GridCell, bool, GridCellHash> closed_set;
  
  // Track g_cost for each node
  std::unordered_map<GridCell, double, GridCellHash> g_score;
  
  // Track parent of each node
  std::unordered_map<GridCell, GridCell, GridCellHash> came_from;
  
  // Initialize start node
  Node start_node;
  start_node.cell = start;
  start_node.g_cost = 0.0;
  start_node.h_cost = calculateHeuristic(start, goal);
  start_node.f_cost = start_node.g_cost + start_node.h_cost;
  start_node.parent = start;
  
  open_set.push(start_node);
  g_score[start] = 0.0;
  
  while (!open_set.empty()) {
    // Get node with lowest f_cost
    Node current = open_set.top();
    open_set.pop();
    
    // Check if we reached the goal
    if (current.cell == goal) {
      return reconstructPath(came_from, start, goal);
    }
    
    // Skip if already processed
    if (closed_set[current.cell]) {
      continue;
    }
    
    closed_set[current.cell] = true;
    
    // Check all neighbors
    std::vector<GridCell> neighbors = getNeighbors(current.cell);
    
    for (const auto& neighbor : neighbors) {
      // Skip if already processed
      if (closed_set[neighbor]) {
        continue;
      }
      
      // Calculate tentative g_cost
      double dx = static_cast<double>(neighbor.x - current.cell.x);
      double dy = static_cast<double>(neighbor.y - current.cell.y);
      double movement_cost = std::sqrt(dx * dx + dy * dy);

      double obstacle_penalty = calculateObstacleCost(neighbor);
      double tentative_g = current.g_cost + movement_cost + obstacle_penalty;
      
      // Check if this path is better
      auto it = g_score.find(neighbor);
      if (it == g_score.end() || tentative_g < it->second) {
        // This path is better, record it
        came_from[neighbor] = current.cell;
        g_score[neighbor] = tentative_g;
        
        Node neighbor_node;
        neighbor_node.cell = neighbor;
        neighbor_node.g_cost = tentative_g;
        neighbor_node.h_cost = calculateHeuristic(neighbor, goal);
        neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost;
        neighbor_node.parent = current.cell;
        
        open_set.push(neighbor_node);
      }
    }
  }
  
  std::cerr << "No path found!" << std::endl;
  return empty_path;
}

}  // namespace astar_planner
