cmake_minimum_required(VERSION 3.14)
project(ChessEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Configurable ONNX Runtime path
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation")
if(NOT ONNXRUNTIME_ROOT)
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "" FORCE)
    else()
        # Search common installation locations
        find_path(ONNXRUNTIME_ROOT
            NAMES include/onnxruntime_cxx_api.h
            PATHS "C:/Program Files/onnxruntime"
                  "C:/onnxruntime"
                  "/usr/local/onnxruntime"
                  "/opt/onnxruntime"
            DOC "Default search paths for ONNX Runtime"
        )
    endif()
endif()

# Validate ONNX Runtime installation
if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT not set! "
            "Set via -DONNXRUNTIME_ROOT=<path> or environment variable.")
endif()

foreach(DIR IN ITEMS "include" "lib")
    if(NOT EXISTS "${ONNXRUNTIME_ROOT}/${DIR}")
        message(FATAL_ERROR "ONNXRUNTIME_ROOT missing required directory: ${ONNXRUNTIME_ROOT}/${DIR}")
    endif()
endforeach()

# Find header location
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH
    REQUIRED
)

# Find library
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

# Create imported target
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
    IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

# Raylib configuration
set(BUILD_EXAMPLES OFF CACHE BOOL "")
set(BUILD_GAMES OFF CACHE BOOL "")
set(USE_AUDIO OFF CACHE BOOL "")
add_subdirectory(third_party/raylib)

# Executable setup
set(SOURCES
    main.cpp
    src/evaluator.cpp
    src/chess_alg.cpp
    src/board_wrapper.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

# Moved AFTER add_executable
target_include_directories(${PROJECT_NAME} PRIVATE
    src
    SYSTEM
        third_party/chess
        third_party/imgui
        third_party/rlimGui
        ${ONNXRUNTIME_INCLUDE_DIR} 
)

target_sources(${PROJECT_NAME} PRIVATE
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/imgui_demo.cpp

    third_party/rlImGui/rlImGui.cpp
)

target_link_libraries(${PROJECT_NAME}
    raylib
    onnxruntime
)

# Windows-specific configuration
if(WIN32)
    # Reduce executable size
    add_compile_options(/Gy /Gw /Os)
    
    # Security hardening
    add_compile_definitions(
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
    )
    
    # Copy ONNX Runtime DLL to output directory
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMENT "Copying ONNX Runtime DLL to output directory"
    )

    # Set entry point to main() and keep console window
    set_target_properties(${PROJECT_NAME} PROPERTIES
        LINK_FLAGS "/SUBSYSTEM:CONSOLE"
    )
endif()

# Platform-specific linking
if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} m dl pthread rt X11 Xi Xrandr GL GLU)
elseif(APPLE)
    find_library(COCOA_LIBRARY Cocoa)
    find_library(OPENGL_LIBRARY OpenGL)
    find_library(IOKIT_LIBRARY IOKit)
    find_library(COREVIDEO_LIBRARY CoreVideo)
    target_link_libraries(${PROJECT_NAME}
        ${COCOA_LIBRARY}
        ${OPENGL_LIBRARY}
        ${IOKIT_LIBRARY}
        ${COREVIDEO_LIBRARY}
    )
endif()