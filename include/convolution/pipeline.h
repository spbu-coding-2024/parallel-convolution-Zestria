#ifndef CONVOLUTION_PIPELINE_H
#define CONVOLUTION_PIPELINE_H

#include <convolution/convolution.h>

/*
 * Fill in image `index` from the caller's source. `in` is a reusable slot owned
 * by the library and holds `opts->width` x `opts->height` pixels.
 */
typedef conv_status (*conv_pipeline_read_fn)(void *ctx, int index, conv_image *in);

/*
 * Store the convolved image for `index`. `out` is a reusable slot owned by the
 * library and is only valid for the duration of the call.
 */
typedef conv_status (*conv_pipeline_write_fn)(void *ctx, int index, const conv_image *out);

typedef struct {
    int width;
    int height;
    int window;
    int threads;
    const conv_kernel *kernel;
    conv_border border;
    const conv_parallel_opts *parallel;
} conv_pipeline_opts;

/*
 * Run a bounded reader -> convolution -> writer pipeline over `count` images
 * using OpenMP tasks.
 *
 * Returns the first callback or convolution failure and stops scheduling further
 * work; `count == 0` is a successful no-op. Invalid arguments return an error code;
 * `threads <= 0` uses the OpenMP default.
 */
conv_status conv_pipeline_run(int count, const conv_pipeline_opts *opts,
                              conv_pipeline_read_fn read,
                              conv_pipeline_write_fn write,
                              void *ctx);

#endif /* CONVOLUTION_PIPELINE_H */
