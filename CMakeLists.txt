cmake_minimum_required(VERSION 3.10)
project(BoVW_Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/libs/opencv/build")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SOURCES "src/*.cpp")
add_executable(bovw ${SOURCES})
target_link_libraries(bovw ${OpenCV_LIBS})

# Create directories
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data/features)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data/dictionary)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data/histograms)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results/html)