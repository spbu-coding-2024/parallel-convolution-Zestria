add_library(image_conv_lib)

target_sources(image_conv_lib
    PRIVATE
        sequential/convolution.c 
        kernels.c
)

target_include_directories(image_conv_lib
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/include 
)

add_executable(image_conv_util)

target_sources(image_conv_util
    PRIVATE
        io/stb_io.c        
)

target_include_directories(image_conv_util
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(image_conv_util
    PRIVATE 
        image_conv_lib
)
