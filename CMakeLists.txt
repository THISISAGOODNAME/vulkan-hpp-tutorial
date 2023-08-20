cmake_minimum_required(VERSION 3.26)
project(vulkan_hpp_tutorial)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)
add_subdirectory(3rd_party)

add_subdirectory(ch0_devenv)
