#ifndef CONVOLUTION_CONVOLUTION_H
#define CONVOLUTION_CONVOLUTION_H

#include <convolution/image.h>

typedef enum {
    CONV_OK = 0,
    CONV_ERR_NULL_ARG,
    CONV_ERR_INVALID_SIZE,
    CONV_ERR_KERNEL_SIZE,
    CONV_ERR_INVALID_PARAM,
    CONV_ERR_IO_READ,
    CONV_ERR_IO_WRITE,
    CONV_ERR_ALLOC_FAIL
} conv_status;

typedef struct {
    int width;
    int height;
    double *data;
    double factor;
    double bias;
} conv_kernel;

conv_status conv_apply_gray(conv_image *in, conv_image *out, conv_kernel *k);

#endif /* CONVOLUTION_CONVOLUTION_H */

