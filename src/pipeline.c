#include <omp.h>
#include <stdlib.h>

#include <convolution/convolution.h>
#include <convolution/pipeline.h>

static void destroy_slots(conv_image **in_slots, conv_image **out_slots, int *token, int window) {
    for (int i = 0; i < window; ++i) {
        conv_image_destroy(in_slots[i]);
        conv_image_destroy(out_slots[i]);
    }
    free(in_slots);
    free(out_slots);
    free(token);
}

static void record_failure(conv_status *status, conv_status st) {
    if (st == CONV_OK) {
        return;
    }
#pragma omp critical
    {
        if (*status == CONV_OK) {
            *status = st;
        }
    }
}

static conv_status convolve_slot(const conv_pipeline_opts *opts, conv_image *in, conv_image *out) {
    conv_status status = CONV_OK;
    if (opts->parallel == NULL) {
        status = conv_apply_gray_border(in, out, opts->kernel, opts->border);
    } else {
        status = conv_apply_gray_parallel(in, out, opts->kernel, opts->border, opts->parallel);
    }
    return status;
}

conv_status conv_pipeline_run(int count, const conv_pipeline_opts *opts,
                              conv_pipeline_read_fn read,
                              conv_pipeline_write_fn write,
                              void *ctx) {
    if (opts == NULL || read == NULL || write == NULL || opts->kernel == NULL) {
        return CONV_ERR_NULL_ARG;
    }
    if (opts->width <= 0 || opts->height <= 0) {
        return CONV_ERR_INVALID_SIZE;
    }
    if (count < 0 || opts->window <= 0) {
        return CONV_ERR_INVALID_PARAM;
    }
    if (count == 0) {
        return CONV_OK;
    }

    int levels = omp_get_max_active_levels();
    omp_set_max_active_levels(2);

    int window = opts->window;
    int threads = opts->threads > 0 ? opts->threads : omp_get_max_threads();

    conv_image **in_slots = calloc((size_t)window, sizeof(*in_slots));
    conv_image **out_slots = calloc((size_t)window, sizeof(*out_slots));
    int *token = calloc((size_t)window, sizeof(*token));
    if (in_slots == NULL || out_slots == NULL || token == NULL) {
        free(in_slots);
        free(out_slots);
        free(token);
        return CONV_ERR_ALLOC_FAIL;
    }

    for (int i = 0; i < window; ++i) {
        in_slots[i] = conv_image_create(opts->width, opts->height);
        out_slots[i] = conv_image_create(opts->width, opts->height);
        if (in_slots[i] == NULL || out_slots[i] == NULL) {
            destroy_slots(in_slots, out_slots, token, window);
            return CONV_ERR_ALLOC_FAIL;
        }
    }

    conv_status status = CONV_OK;

#pragma omp parallel default(none) num_threads(threads) \
    shared(count, window, opts, in_slots, out_slots, token, read, write, ctx, status)
    {
#pragma omp single
        for (int i = 0; i < count; ++i) {
            int stop;
#pragma omp critical
            stop = (status != CONV_OK);
            if (stop) {
                break;
            }

            int slot = i % window;

#pragma omp task depend(inout: token[slot])
            record_failure(&status, read(ctx, i, in_slots[slot]));

#pragma omp task depend(inout: token[slot])
            record_failure(&status, convolve_slot(opts, in_slots[slot], out_slots[slot]));

#pragma omp task depend(inout: token[slot])
            record_failure(&status, write(ctx, i, out_slots[slot]));
        }
    }

    destroy_slots(in_slots, out_slots, token, window);
    omp_set_max_active_levels(levels);

    return status;
}
