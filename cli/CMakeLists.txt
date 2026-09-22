add_executable(image_conv_cli)

target_sources(image_conv_cli
    PRIVATE
        main.c      
)

target_include_directories(image_conv_cli
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party
)

target_link_libraries(image_conv_cli
    PRIVATE 
        image_conv_lib
)

