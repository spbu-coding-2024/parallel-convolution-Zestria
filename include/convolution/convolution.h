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
 * A kernel is valid when its width and height are positive and odd, so the
 * kernel has a well-defined center, and its coefficients are not NULL.
 *
 * Returns CONV_ERR_NULL_ARG for a NULL kernel or NULL coefficients,
 * CONV_ERR_INVALID_SIZE for a non-positive width or height, and
 * CONV_ERR_KERNEL_SIZE for an even width or height; otherwise CONV_OK.
 */
conv_status conv_kernel_validate(const conv_kernel *kernel);

typedef enum {
    CONV_BORDER_WRAP,   /* modulo: -1 -> limit-1, limit -> 0            */
    CONV_BORDER_CLAMP,  /* repeat edge: -1 -> 0, limit -> limit-1       */
    CONV_BORDER_ZERO,   /* samples outside the image count as 0         */
    CONV_BORDER_MIRROR, /* reflect w/o edge: -1 -> 1, limit -> limit-2  */
} conv_border;

/*
 * Correlate `in` with `k` and write the result to `out` using different border strategy.
 *
 * In-place application is rejected (CONV_ERR_INVALID_PARAM): `in` and `out`
 * must use different pixel buffers.
 */
conv_status conv_apply_gray_border(const conv_image *in, conv_image *out,
                                   const conv_kernel *k, conv_border border);

/*
 * Correlate `in` with `k` and write the result to `out` using wrap around strategy.
 *
 * In-place application is rejected (CONV_ERR_INVALID_PARAM): `in` and `out`
 * must use different pixel buffers.
 */
conv_status conv_apply_gray(const conv_image *in, conv_image *out, const conv_kernel *k);

#endif /* CONVOLUTION_CONVOLUTION_H */

