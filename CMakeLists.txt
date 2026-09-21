cmake_minimum_required (VERSION 3.10)

project(conv_project C)

# .clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)

# tests
option(BUILD_TESTING "Enable testing and build tests" ON)

if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

