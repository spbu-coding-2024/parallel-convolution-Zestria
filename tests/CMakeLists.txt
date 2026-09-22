add_executable(test_seq_conv_lib)

target_sources(test_seq_conv_lib
    PRIVATE
        sequential/test_convolution.c
)

target_link_libraries(test_seq_conv_lib
    PRIVATE
        image_conv_lib
)

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

add_test(
    NAME test_image
    COMMAND test_image
)

