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
    0, 0, 0,
    0, 1, 0,
    0, 0, 0
};

static double KERNEL_BLUR_3x3_DATA[9] = {
    0.0, 0.2,  0.0,
    0.2, 0.2,  0.2,
    0.0, 0.2,  0.0
};

static double KERNEL_BLUR_5x5_DATA[25] = {
    0, 0, 1, 0, 0,
    0, 1, 1, 1, 0,
    1, 1, 1, 1, 1,
    0, 1, 1, 1, 0,
    0, 0, 1, 0, 0,
};

static double KERNEL_GAUSSIAN_BLUR_3x3_DATA[9] = {
    0.077847, 0.123317, 0.077847,
    0.123317, 0.195346, 0.123317,
    0.077847, 0.123317, 0.077847,
};

static double KERNEL_MOTION_BLUR_9x9_DATA[81] = {
    1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1,
};

static double KERNEL_FIND_HORIZONTAL_EDGES_5x5_DATA[25] = {
    0,  0, -1,  0,  0,
    0,  0, -1,  0,  0,
    0,  0,  2,  0,  0,
    0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,
};

static double KERNEL_FIND_VERTICAL_EDGES_5x5_DATA[25] = {
    0,  0, -1,  0,  0,
    0,  0, -1,  0,  0,
    0,  0,  4,  0,  0,
    0,  0, -1,  0,  0,
    0,  0, -1,  0,  0,
};

static double KERNEL_FIND_DIAGONAL_EDGES_5x5_DATA[25] = {
    -1,  0,  0,  0,  0,
    0, -2,  0,  0,  0,
    0,  0,  6,  0,  0,
    0,  0,  0, -2,  0,
    0,  0,  0,  0, -1,
};

static double KERNEL_FIND_ALL_EDGES_3x3_DATA[9] = {
    -1, -1, -1,
    -1,  8, -1,
    -1, -1, -1
};

static double KERNEL_SHARPEN_3x3_DATA[9] = {
    -1, -1, -1,
    -1,  9, -1,
    -1, -1, -1
};

static double KERNEL_SHARPEN_5x5_DATA[25] = {
    -1, -1, -1, -1, -1,
    -1,  2,  2,  2, -1,
    -1,  2,  8,  2, -1,
    -1,  2,  2,  2, -1,
    -1, -1, -1, -1, -1,
};

static double KERNEL_SHARPEN_EXCESSIVELY_3x3_DATA[9] = {
    1,  1,  1,
    1, -7,  1,
    1,  1,  1
};

static double KERNEL_EMBOSS_3x3_DATA[9] = {
    -1, -1,  0,
    -1,  0,  1,
    0,  1,  1
};

static double KERNEL_EMBOSS_5x5_DATA[25] = {
    -1, -1, -1, -1,  0,
    -1, -1, -1,  0,  1,
    -1, -1,  0,  1,  1,
    -1,  0,  1,  1,  1,
    0,  1,  1,  1,  1
};

static double KERNEL_MEAN_3x3_DATA[9] = {
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
};

const conv_kernel KERNEL_IDENTITY_3x3 = {
    .width = 3,
    .height = 3,
    .data = KERNEL_IDENTITY_3x3_DATA,
    .factor = 1.0,
    .bias = 0.0
};

const conv_kernel KERNEL_BLUR_3x3 = {
    .width = 3,
    .height = 3,
    .data = KERNEL_BLUR_3x3_DATA,
    .factor = 1.0,
    .bias = 0.0
};

const conv_kernel KERNEL_BLUR_5x5 = {
    .width = 5,
    .height = 5,
    .data = KERNEL_BLUR_5x5_DATA,
    .factor = 1.0 / 13.0,
    .bias = 0.0
};

const conv_kernel KERNEL_GAUSSIAN_BLUR_3x3 = {
    .width = 3,
    .height = 3,
    .data = KERNEL_GAUSSIAN_BLUR_3x3_DATA,
    .factor = 1.0,
    .bias = 0.0
};

const conv_kernel KERNEL_MOTION_BLUR_9x9 = {
    .width = 9,
    .height = 9,
    .data = KERNEL_MOTION_BLUR_9x9_DATA,
    .factor = 1.0 / 9.0,
    .bias = 0.0
};

const conv_kernel KERNEL_FIND_HORIZONTAL_EDGES_5x5 = {
    .width = 5,
    .height = 5,
    .data = KERNEL_FIND_HORIZONTAL_EDGES_5x5_DATA,
    .factor = 1.0,
    .bias = 0.0
};

const conv_kernel KERNEL_FIND_VERTICAL_EDGES_5x5 = {
    .width = 5,
    .height = 5,
    .data = KERNEL_FIND_VERTICAL_EDGES_5x5_DATA,
    .factor = 1.0,
    .bias = 0.0
};

const conv_kernel KERNEL_FIND_DIAGONAL_EDGES_5x5 = {
    .width = 5,
    .height = 5,
    .data = KERNEL_FIND_DIAGONAL_EDGES_5x5_DATA,
    .factor = 1.0,
    .bias = 0.0
};

const conv_kernel KERNEL_FIND_ALL_EDGES_3x3 = {
    .width = 3,
    .height = 3,
    .data = KERNEL_FIND_ALL_EDGES_3x3_DATA,
    .factor = 1.0,
    .bias = 0.0
};

const conv_kernel KERNEL_SHARPEN_3x3 = {
    .width = 3,
    .height = 3,
    .data = KERNEL_SHARPEN_3x3_DATA,
    .factor = 1.0,
    .bias = 0.0
};

const conv_kernel KERNEL_SHARPEN_5x5 = {
    .width = 5,
    .height = 5,
    .data = KERNEL_SHARPEN_5x5_DATA,
    .factor = 1.0 / 8.0,
    .bias = 0.0
};

const conv_kernel KERNEL_SHARPEN_EXCESSIVELY_3x3 = {
    .width = 3,
    .height = 3,
    .data = KERNEL_SHARPEN_EXCESSIVELY_3x3_DATA,
    .factor = 1.0,
    .bias = 0.0
};

const conv_kernel KERNEL_EMBOSS_3x3 = {
    .width = 3,
    .height = 3,
    .data = KERNEL_EMBOSS_3x3_DATA,
    .factor = 1.0,
    .bias = 128.0
};

const conv_kernel KERNEL_EMBOSS_5x5 = {
    .width = 5,
    .height = 5,
    .data = KERNEL_EMBOSS_5x5_DATA,
    .factor = 1.0,
    .bias = 128.0
};

const conv_kernel KERNEL_MEAN_3x3 = {
    .width = 3,
    .height = 3,
    .data = KERNEL_MEAN_3x3_DATA,
    .factor = 1.0 / 9.0,
    .bias = 0.0
};
