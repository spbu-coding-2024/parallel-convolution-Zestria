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

/*
 * Correlate `in` with `k` and write the result to `out` using wrap around strategy.
 *
 * In-place application is rejected (CONV_ERR_INVALID_PARAM): `in` and `out`
 * must use different pixel buffers.
 */
conv_status conv_apply_gray(const conv_image *in, conv_image *out, const conv_kernel *k);

#endif /* CONVOLUTION_CONVOLUTION_H */

