cmake_minimum_required(VERSION 3.26)
project(vulkan_hpp_tutorial)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)
add_subdirectory(3rd_party)

# demos
add_subdirectory(ch0_devenv)
add_subdirectory(ch1_triangle)
add_subdirectory(ch2_vertexbuffer)
add_subdirectory(ch3_uniformbuffer)
add_subdirectory(ch4_texturemapping)
