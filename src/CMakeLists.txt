add_library(image_conv_lib)

target_sources(image_conv_lib
    PRIVATE
        sequential/convolution.c
        image.c
        kernels.c
)

target_include_directories(image_conv_lib
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/include 
)

target_link_libraries(image_conv_lib
    PUBLIC
        OpenMP::OpenMP_C
)

target_compile_options(image_conv_lib
    PRIVATE
        -Wall -Wextra
)

