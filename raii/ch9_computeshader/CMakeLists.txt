set(APP_NAME ch9_computeshader_raii)
add_executable(${APP_NAME} ch9_computeshader.cpp)
target_link_libraries(${APP_NAME} PRIVATE
        glfw
        glm
        Vulkan::Vulkan
)

# shader compile
if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator")
else()
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin32/glslangValidator")
endif()

set(GLSL_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader_base.frag
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader_base.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader_base.comp)

foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.spv")
    add_custom_command(
            TARGET ${APP_NAME} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders/"
            COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
            DEPENDS ${GLSL}
            VERBATIM
    )
endforeach(GLSL)

