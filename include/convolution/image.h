#ifndef CONVOLUTION_IMAGE_H
#define CONVOLUTION_IMAGE_H

#include <stdint.h>

typedef struct {
    int width;
    int height;
    uint8_t *data;
} conv_image;

#endif /* CONVOLUTION_IMAGE_H */

