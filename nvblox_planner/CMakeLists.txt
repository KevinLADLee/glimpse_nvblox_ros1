# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.10)
#add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(nvblox_planar LANGUAGES CXX CUDA)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Default to release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message( STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )

################
# DEPENDENCIES #
################

list(APPEND CMAKE_INCLUDE_PATH "/usr/local/include")

find_package (Eigen3 3.3.7 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  std_srvs
  sensor_msgs
  geometry_msgs
  nvblox_msgs
  tf2_ros
  tf2_eigen
  cv_bridge
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS
  include
  LIBRARIES
  ${PROJECT_NAME}
  3rdparty
  CATKIN_DEPENDS
    roscpp
    std_msgs
    std_srvs
    sensor_msgs
    geometry_msgs
    tf2_ros
    tf2_eigen
    nvblox_msgs
)

# Process nvblox
set(NVBLOX_DIR ../nvblox/nvblox)

# Disable tests and experiments
set(BUILD_EXPERIMENTS OFF)
set(BUILD_TESTS OFF)

# Direct the child cmake to install eigen to the devel space.
# Note that we put it in the include/nvblox_ros directory, which should stop it
# colliding with other versions of eigen used by other catkin packages.
set(NVBLOX_EIGEN_CATKIN_INCLUDE_DESTINATION "${CMAKE_INSTALL_PREFIX}")
set(EIGEN_INCLUDE_DESTINATION "${NVBLOX_EIGEN_CATKIN_INCLUDE_DESTINATION}")
set(STDGPU_INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX})

########
# CUDA #
########
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcudafe --display_error_number --disable-warnings ")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC")

##########
# NVBLOX #
##########
#add_subdirectory(${NVBLOX_DIR} nvblox)

include_directories(AFTER include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
include_directories(${NVBLOX_DIR}/include
                    ${NVBLOX_DIR}/executables/include 
                    ${NVBLOX_DIR}/experiments/include)

message("${Magenta}INFO: ${CMAKE_MODULE_PATH}.${ColourReset}")
message("${Magenta}INFO: ${CMAKE_PREFIX_PATH}.${ColourReset}")
message("${Magenta}INFO: ${CMAKE_FIND_ROOT_PATH}.${ColourReset}")
message("${Magenta}INFO: ${CMAKE_INCLUDE_PATH}.${ColourReset}")

#############
# LIBRARIES #
#############
# add_library(${PROJECT_NAME}_lib SHARED
#   src/conversions.cpp
#   src/cuda/conversions.cu
#   src/transformer.cpp
# )

# target_link_libraries(${PROJECT_NAME}_lib 
#   nvblox_interface
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES})

# add_dependencies(${PROJECT_NAME}_lib
#   ${catkin_EXPORTED_TARGETS})

# target_include_directories(${PROJECT_NAME}_lib SYSTEM PUBLIC
#   ${catkin_INCLUDE_DIRS})  

############
# BINARIES #
############
# nvblox_node
# add_executable(nvblox_node src/nvblox_node.cpp)
# target_link_libraries(nvblox_node ${PROJECT_NAME}_lib)
# add_dependencies(nvblox_node ${catkin_EXPORTED_TARGETS})
# target_include_directories(nvblox_node SYSTEM PUBLIC
#   ${catkin_INCLUDE_DIRS}
# )


