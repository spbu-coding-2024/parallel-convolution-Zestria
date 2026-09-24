#include <convolution/kernels.h>

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

