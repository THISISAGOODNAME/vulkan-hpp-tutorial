add_executable(ch0_devenv ch0_devenv.cpp)
target_link_libraries(ch0_devenv PRIVATE
        glfw
        glm
        Vulkan::Vulkan
)