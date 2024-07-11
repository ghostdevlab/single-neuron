cmake_minimum_required(VERSION 3.26)
project(Neuron)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_RPATH "/Library/Frameworks/")

find_package(SDL2_IMAGE REQUIRED)
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

add_executable(Neuron
        main_training.cpp)
target_link_libraries(Neuron ${SDL2_LIBRARIES} SDL2_image::SDL2_image)