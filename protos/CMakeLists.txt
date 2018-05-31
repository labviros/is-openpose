cmake_minimum_required(VERSION 3.0)
project(openpose-msgs C CXX)

include(GNUInstallDirs)

# Protobuf
find_package(Protobuf 3 REQUIRED)

file(GLOB protos "*.proto")
foreach(proto ${protos})
  PROTOBUF_GENERATE_CPP(SRC HEADER ${proto})
  set(PROTO_SRCS "${PROTO_SRCS};${SRC}")
  set(PROTO_HEADERS "${PROTO_HEADERS};${HEADER}")
endforeach()

add_library(openpose-msgs ${PROTO_SRCS})
set_property(TARGET openpose-msgs PROPERTY CXX_STANDARD 11)
target_link_libraries(
  openpose-msgs
 PUBLIC 
  ${PROTOBUF_LIBRARIES}
)

target_include_directories(
  openpose-msgs 
 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${PROTOBUF_INCLUDE_DIRS}
)

install(
  TARGETS openpose-msgs
  EXPORT openposeMsgsTargets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

#install headers
install(
  FILES ${PROTO_HEADERS}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/is/openpose/msgs/"
)

#install protos
install(
  FILES ${protos}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/is/openpose/msgs/"
)

#install config
install(
  EXPORT openposeMsgsTargets
  FILE openpose-msgsConfig.cmake
  NAMESPACE openpose-msgs::
  DESTINATION lib/cmake/openpose-msgs
)

# register project in CMake user registry
export(PACKAGE ${PROJECT_NAME})

add_library(openpose-msgs::openpose-msgs ALIAS openpose-msgs)