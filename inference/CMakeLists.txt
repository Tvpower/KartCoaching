cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(inference)

set(CMAKE_PREFIX_PATH /opt/libtorch)

set(TORCH_CUDA_ARCH_LIST "12.0")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

add_executable(inference
        main.cpp
        inference.h
        inference.cpp)
target_link_libraries(inference
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS})

set_property(TARGET inference PROPERTY CXX_STANDARD 17)

