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

/*
 * Apply `kernels[0]` .. `kernels[count - 1]` to `in` from left to right and
 * write the final result to `out`.
 *
 * `count` must be at least 1 and `kernels` must hold `count` valid kernels.
 * In-place application is rejected (CONV_ERR_INVALID_PARAM): `in` and `out`
 * must use different pixel buffers.
 */
conv_status conv_apply_gray_chain(const conv_image *in, conv_image *out,
                                  const conv_kernel *const *kernels, int count,
                                  conv_border border);

typedef enum {
    CONV_PART_PIXEL,
    CONV_PART_ROW,
    CONV_PART_COLUMN,
    CONV_PART_TILE,
} conv_partition;

typedef enum {
    CONV_SCHED_STATIC,
    CONV_SCHED_DYNAMIC,
    CONV_SCHED_GUIDED,
    CONV_SCHED_RUNTIME,
} conv_omp_schedule;

typedef struct {
    int threads;
    conv_partition partition;
    conv_omp_schedule schedule;
    int chunk;
    int tile_w;
    int tile_h;
} conv_parallel_opts;

/*
 * Parallel correlate `in` with `k` into `out`, byte-for-byte identical to
 * conv_apply_gray_border() for the same border mode.
 *
 * In-place application is rejected (CONV_ERR_INVALID_PARAM): `in` and `out`
 * must use different pixel buffers.
 */
conv_status conv_apply_gray_parallel(const conv_image *in, conv_image *out,
                                     const conv_kernel *k, conv_border border,
                                     const conv_parallel_opts *opts);

#endif /* CONVOLUTION_CONVOLUTION_H */

