#ifndef ASTAR_PLANNER_ASTAR_HPP_
#define ASTAR_PLANNER_ASTAR_HPP_

#include <vector>
#include <unordered_map>
#include <queue>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace astar_planner
{

struct GridCell
{
  int x;
  int y;

  bool operator==(const GridCell& other) const
  {
    return x == other.x && y == other.y;
  }
};

struct GridCellHash
{
  std::size_t operator()(const GridCell& cell) const
  {
    return std::hash<int>()(cell.x) ^ (std::hash<int>()(cell.y) << 1);
  }
};

struct Node
{
  GridCell cell;
  double g_cost;
  double h_cost;
  double f_cost;
  GridCell parent;

  bool operator>(const Node& other) const
  {
    return f_cost > other.f_cost;
  }
};

class AStar
{
public:
  AStar();
  ~AStar();

  void setMap(const std::vector<std::vector<int>>& map);
  
  // NEW: Method to set robot radius in grid cells
  void setRobotRadius(int radius_in_cells);

  std::vector<GridCell> findPath(const GridCell& start, const GridCell& goal);

private:
  double calculateHeuristic(const GridCell& a, const GridCell& b) const;
  bool isValid(const GridCell& cell) const;
  std::vector<GridCell> getNeighbors(const GridCell& cell) const;
  std::vector<GridCell> reconstructPath(
    const std::unordered_map<GridCell, GridCell, GridCellHash>& came_from,
    const GridCell& start,
    const GridCell& goal) const;

  std::vector<std::vector<int>> map_;
  int map_width_;
  int map_height_;
  
  // NEW: Variable to store robot size
  int robot_radius_; 
};

}  // namespace astar_planner

#endif  // ASTAR_PLANNER_ASTAR_HPP_