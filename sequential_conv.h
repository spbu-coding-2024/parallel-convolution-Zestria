#include "conv_common.h"

#ifndef CONV_TEST_BUILD

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#endif

void convolve_sequential(ConvArgs *a);
