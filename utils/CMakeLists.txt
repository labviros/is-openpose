cmake_minimum_required(VERSION 3.0)

find_package(OpenPose REQUIRED)
find_package(ismsgs REQUIRED)

add_library(utils-skeletons skeletons.cpp)

target_link_libraries(
  utils-skeletons
 PUBLIC
  openpose-msgs::openpose-msgs
  ismsgs::ismsgs
 PRIVATE
  openpose
)

target_include_directories(
  utils-skeletons
 PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>"
)

set_property(TARGET utils-skeletons PROPERTY CXX_STANDARD 14)