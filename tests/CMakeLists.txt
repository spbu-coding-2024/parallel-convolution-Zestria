add_executable(test_seq_conv_lib)

target_sources(test_seq_conv_lib
    PRIVATE
        sequential/test_convolution.c
)

target_link_libraries(test_seq_conv_lib
    PRIVATE
        image_conv_lib
)

target_compile_options(test_seq_conv_lib PRIVATE -Wall -Wextra)

add_test(
    NAME test_seq_conv_lib 
    COMMAND test_seq_conv_lib
)

add_executable(test_image)

target_sources(test_image
    PRIVATE
        test_image.c
)

target_link_libraries(test_image
    PRIVATE
        image_conv_lib
)

target_compile_options(test_image PRIVATE -Wall -Wextra)

add_test(
    NAME test_image
    COMMAND test_image
)

add_executable(test_kernel)

target_sources(test_kernel
    PRIVATE
        test_kernel.c
)

target_link_libraries(test_kernel
    PRIVATE
        image_conv_lib
)

target_compile_options(test_kernel PRIVATE -Wall -Wextra)

add_test(
    NAME test_kernel
    COMMAND test_kernel
)

add_executable(test_cli)

target_sources(test_cli
    PRIVATE
        test_cli.c
)

target_link_libraries(test_cli
    PRIVATE
        image_conv_cli_io
)

target_compile_options(test_cli PRIVATE -Wall -Wextra)

add_test(
    NAME test_cli
    COMMAND test_cli ${CMAKE_SOURCE_DIR}/assets/emmastone.png
)

