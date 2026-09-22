#ifndef CONVOLUTION_IMAGE_H
#define CONVOLUTION_IMAGE_H

#include <stdint.h>

typedef struct {
    int width;
    int height;
    uint8_t *data;
} conv_image;

conv_image *conv_image_create(int width, int height);
void conv_image_destroy(conv_image *image);

#endif /* CONVOLUTION_IMAGE_H */

