cmake_minimum_required(VERSION 2.8.3)
project(grid_submap)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  grid_map_core
  grid_map_msgs
  grid_map_ros
)


catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(grid_submap src/grid_submap.cpp)
target_link_libraries(grid_submap ${catkin_LIBRARIES})