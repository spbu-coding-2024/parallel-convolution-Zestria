#include <stdlib.h>

#include <convolution/kernels.h>

conv_kernel *conv_kernel_create(int width, int height) {
    if (width <= 0 || height <= 0) {
        return NULL;
    }

    conv_kernel *kernel = calloc(1, sizeof(conv_kernel));
    if (kernel == NULL) {
        return NULL;
    }

    kernel->data = calloc((size_t)width * (size_t)height, sizeof(double));
    if (kernel->data == NULL) {
        free(kernel);
        return NULL;
    }

    kernel->width = width;
    kernel->height = height;
    kernel->factor = 1.0;
    kernel->bias = 0.0;

    if (conv_kernel_validate(kernel) != CONV_OK) {
        conv_kernel_destroy(kernel);
        return NULL;
    }
    return kernel;
}

void conv_kernel_destroy(conv_kernel *kernel) {
    if (kernel == NULL) {
        return;
    }

    free(kernel->data);
    free(kernel);
}

conv_status conv_kernel_validate(const conv_kernel *kernel) {
    if (kernel == NULL || kernel->data == NULL) {
        return CONV_ERR_NULL_ARG;
    }
    if (kernel->width <= 0 || kernel->height <= 0) {
        return CONV_ERR_INVALID_SIZE;
    }
    if (kernel->width % 2 == 0 || kernel->height % 2 == 0) {
        return CONV_ERR_KERNEL_SIZE;
    }
    return CONV_OK;
}

static double KERNEL_IDENTITY_3x3_DATA[9] = {
    0,0,0,
    0,1,0,
    0,0,0
};

const conv_kernel KERNEL_IDENTITY_3x3 = {
    .width = 3,
    .height = 3,
    .data = KERNEL_IDENTITY_3x3_DATA,
    .factor = 1.0,
    .bias = 0.0
};

