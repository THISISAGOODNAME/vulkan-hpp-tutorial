set(APP_NAME ch0_devenv_raii)
add_executable(${APP_NAME} ch0_devenv.cpp)
target_link_libraries(${APP_NAME} PRIVATE
        glfw
        glm
        Vulkan::Vulkan
)