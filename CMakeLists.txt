cmake_minimum_required (VERSION 3.10)

project(conv_project C)

# .clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)

add_subdirectory(src)
add_subdirectory(cli)

# tests
option(BUILD_TESTING "Enable testing and build tests" ON)

if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

