#include <stddef.h>
#include <stdlib.h>

#include <convolution/convolution.h>

conv_image *conv_image_create(int width, int height) {
    if (width <= 0 || height <= 0) {
        return NULL; 
    }
    conv_image *image = malloc(sizeof(conv_image));
    
    if (image == NULL) {
        return NULL;
    }

    size_t count = (size_t)width * (size_t)height;
    image->data = calloc(count, sizeof(uint8_t));
    if (image->data == NULL) {
        free(image);
        return NULL;
    }
    image->width = width;
    image->height = height;
    return image;
}

void conv_image_destroy(conv_image *image) {
    if (image == NULL) {
        return;
    }
    free(image->data);
    free(image);
}

