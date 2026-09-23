add_library(image_conv_cli_io STATIC)

target_sources(image_conv_cli_io
    PRIVATE
        io.c
)

target_include_directories(image_conv_cli_io
    PUBLIC
        ${CMAKE_SOURCE_DIR}/cli
)

target_include_directories(image_conv_cli_io
    PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party
)

target_compile_options(image_conv_cli_io PRIVATE -Wall -Wextra)

target_link_libraries(image_conv_cli_io
    PUBLIC
        image_conv_lib
        m
)

add_executable(image_conv_cli)

target_sources(image_conv_cli
    PRIVATE
        main.c
)

target_include_directories(image_conv_cli
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(image_conv_cli PRIVATE -Wall -Wextra)

target_link_libraries(image_conv_cli
    PRIVATE
        image_conv_cli_io
        OpenMP::OpenMP_C
)
