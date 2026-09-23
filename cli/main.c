#include <convolution/convolution.h>
#include <convolution/image.h>
#include <stddef.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

conv_status conv_io_load_png(char *path, conv_image **out) {
    if (out == NULL || path == NULL) {
        return CONV_ERR_NULL_ARG;
    }

    *out = NULL;
    int width = 0;
    int height = 0;
    int channels_in_file = 0;
    int desired_channels = 1;
    
    uint8_t *pixels = stbi_load(path, &width, &height, &channels_in_file, desired_channels);
    if (pixels == NULL) {
        return CONV_ERR_IO_READ;
    }
    
    conv_image *image = conv_image_create(width, height);
    if (image == NULL) {
        stbi_image_free(pixels);
        return CONV_ERR_ALLOC_FAIL;
    }

    size_t count = (size_t)width * (size_t)height;
    for (size_t i = 0; i < count; ++i) {
        image->data[i] = pixels[i];
    }
    stbi_image_free(pixels);

    *out = image;
    return CONV_OK;
}

conv_status conv_io_save_png(char *path, conv_image *image) {
    if (path == NULL || image == NULL || image->data == NULL || image->width <= 0 || image->height <= 0) {
        return CONV_ERR_INVALID_PARAM;
    }
    
    size_t count = (size_t)image->width * (size_t)image->height;
    uint8_t *pixels = malloc(count);
    if (pixels == NULL) {
        return CONV_ERR_ALLOC_FAIL;
    }

    for (size_t i = 0; i < count; ++i) {
        pixels[i] = image->data[i];
    }

    int ok = stbi_write_png(path, image->width, image->height, 1, pixels, image->width);
    free(pixels);
    return ok ? CONV_OK : CONV_ERR_IO_WRITE;
}

int main(void) {
    
}
