cmake_minimum_required(VERSION 3.27)
project(jay)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)

add_executable(jay main.cpp)

target_link_libraries(jay ${OpenCV_LIBS})
