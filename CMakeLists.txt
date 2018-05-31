cmake_minimum_required(VERSION 3.0)
include(GNUInstallDirs)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(protos)
add_subdirectory(utils)

find_package(is REQUIRED)
find_package(ismsgs REQUIRED) #fixme
find_package(prometheus-cpp REQUIRED) #fixme

add_executable(rpc.bin rpc.cpp)
target_link_libraries(
  rpc.bin
 PRIVATE
  is::is
  openpose-msgs::openpose-msgs
  utils-skeletons
)
set_property(TARGET rpc.bin PROPERTY CXX_STANDARD 14)

add_executable(stream.bin stream.cpp)
target_link_libraries(
  stream.bin
 PRIVATE
  is::is
  openpose-msgs::openpose-msgs
  utils-skeletons
)
set_property(TARGET stream.bin PROPERTY CXX_STANDARD 14)

install(
  TARGETS rpc.bin stream.bin
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)