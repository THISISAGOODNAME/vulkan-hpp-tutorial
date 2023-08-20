add_executable(ch1_devenv ch1_triangle.cpp)
target_link_libraries(ch1_devenv PRIVATE
        glfw
        glm
        Vulkan::Vulkan
)