#ifndef TEST_DATA_H
#define TEST_DATA_H

#include <stdint.h>

#include <convolution/image.h>

/*
 * Fill an image with a deterministic gradient.
 */
static inline void fill_gradient(conv_image *image) {
    for (int y = 0; y < image->height; ++y) {
        for (int x = 0; x < image->width; ++x) {
            image->data[y * image->width + x] =
                (uint8_t)((x * 7 + y * 13 + 3) % 256);
        }
    }
}

#endif /* TEST_DATA_H */
