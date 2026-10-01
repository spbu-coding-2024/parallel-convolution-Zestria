#include <stddef.h>
#include <omp.h>

#include <convolution/convolution.h>

static uint8_t clamp_u8(double v) {
    if (v < 0.0) {
        return 0;
    }
    if (v > 255.0) {
        return 255;
    }
    return (uint8_t)v;
}

static conv_status validate(const conv_image *in, conv_image *out, const conv_kernel *k) {
    if (in == NULL || out == NULL || in->data == NULL || out->data == NULL) {
        return CONV_ERR_NULL_ARG;
    }
    if (in->width <= 0 || in->height <= 0 || out->width != in->width || out->height != in->height) {
        return CONV_ERR_INVALID_SIZE;
    }

    conv_status kernel_status = conv_kernel_validate(k);
    if (kernel_status != CONV_OK) {
        return kernel_status;
    }

    if (k->width > in->width || k->height > in->height) {
        return CONV_ERR_KERNEL_SIZE;
    }
    if (in->data == out->data) {
        return CONV_ERR_INVALID_PARAM;
    }
    return CONV_OK;
}

static int border_index(int i, int limit, conv_border mode) {
    if (i >= 0 && i < limit) {
        return i;
    }

    switch (mode) {
    case CONV_BORDER_WRAP:
        return (i % limit + limit) % limit;
    case CONV_BORDER_CLAMP:
        return i < 0 ? 0 : limit - 1;
    case CONV_BORDER_ZERO:
        return -1;
    case CONV_BORDER_MIRROR:
        if (limit == 1) {
            return 0;
        }
        while (i < 0 || i >= limit) {
            i = i < 0 ? -i : 2 * limit - 2 - i;
        }
        return i;
    }

    return -1;
}

static uint8_t convolve_pixel(const conv_image *in, const conv_kernel *k,
                              conv_border border, int x, int y) {
    int w = in->width;
    int h = in->height;
    int kw = k->width;
    int kh = k->height;
    int dx = kw / 2;
    int dy = kh / 2;
    double acc = 0.0;

    for (int ky = 0; ky < kh; ++ky) {
    for (int kx = 0; kx < kw; ++kx) {
        int iy = border_index(y - dy + ky, h, border);
        int ix = border_index(x - dx + kx, w, border);

        if (iy < 0 || ix < 0) {
            continue; /* CONV_BORDER_ZERO case */
        }

        acc += in->data[iy * w + ix] * k->data[ky * kw + kx];
    }
    }

    return clamp_u8(k->factor * acc + k->bias);
}

static void convolve_pixels(const conv_image *in, conv_image *out, const conv_kernel *k,
                            conv_border border, int threads) {
    int w = in->width;
    int total = w * in->height;

#pragma omp parallel for schedule(runtime) default(none) \
    shared(in, out, k, border, w, total, threads) num_threads(threads)
    for (int idx = 0; idx < total; ++idx) {
        out->data[idx] = convolve_pixel(in, k, border, idx % w, idx / w);
    }
}

static void convolve_rows(const conv_image *in, conv_image *out, const conv_kernel *k,
                          conv_border border, int threads) {
    int w = in->width;
    int h = in->height; 

#pragma omp parallel for schedule(runtime) default(none) \
    shared(in, out, k, border, w, h, threads) num_threads(threads)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            out->data[y * w + x] = convolve_pixel(in, k, border, x, y);
        }
    }
}

static void convolve_columns(const conv_image *in, conv_image *out, const conv_kernel *k,
                             conv_border border, int threads) {
    int w = in->width;
    int h = in->height;

#pragma omp parallel for schedule(runtime) default(none) \
    shared(in, out, k, border, w, h, threads) num_threads(threads)
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            out->data[y * w + x] = convolve_pixel(in, k, border, x, y);
        }
    }
}

static void convolve_tiles(const conv_image *in, conv_image *out, const conv_kernel *k,
                           conv_border border, int tile_w, int tile_h, int threads) {
    int w = in->width;
    int h = in->height;
    int tiles_x = (w + tile_w - 1) / tile_w;
    int tiles_y = (h + tile_h - 1) / tile_h;
    int count = tiles_x * tiles_y;

#pragma omp parallel for schedule(runtime) default(none) \
    shared(in, out, k, border, w, h, tiles_x, tile_w, tile_h, count, threads) \
    num_threads(threads)
    for (int t = 0; t < count; ++t) {
        int tx = (t % tiles_x) * tile_w;
        int ty = (t / tiles_x) * tile_h;
        int x_end = tx + tile_w < w ? tx + tile_w : w;
        int y_end = ty + tile_h < h ? ty + tile_h : h;

        for (int y = ty; y < y_end; ++y) {
            for (int x = tx; x < x_end; ++x) {
                out->data[y * w + x] = convolve_pixel(in, k, border, x, y);
            }
        }
    }
}

static void set_schedule(conv_omp_schedule schedule, int chunk) {
    switch (schedule) {
    case CONV_SCHED_STATIC:
        omp_set_schedule(omp_sched_static, chunk);
        break;
    case CONV_SCHED_DYNAMIC:
        omp_set_schedule(omp_sched_dynamic, chunk);
        break;
    case CONV_SCHED_GUIDED:
        omp_set_schedule(omp_sched_guided, chunk);
        break;
    case CONV_SCHED_RUNTIME:
        break;
    }
}

conv_status conv_apply_gray_parallel(const conv_image *in, conv_image *out,
                                     const conv_kernel *k, conv_border border,
                                     const conv_parallel_opts *opts) {
    conv_status st = validate(in, out, k);
    if (st != CONV_OK) {
        return st;
    }

    static const conv_parallel_opts defaults = {
        0, CONV_PART_PIXEL, CONV_SCHED_STATIC, 0, 0, 0
    };
    if (opts == NULL) {
        opts = &defaults;
    }

    int threads = opts->threads > 0 ? opts->threads : omp_get_max_threads();
    int chunk = opts->chunk > 0 ? opts->chunk : 0;
    int tile_w = opts->tile_w > 0 ? opts->tile_w : 16;
    int tile_h = opts->tile_h > 0 ? opts->tile_h : 16;

    set_schedule(opts->schedule, chunk);

    switch (opts->partition) {
    case CONV_PART_PIXEL:
        convolve_pixels(in, out, k, border, threads);
        break;
    case CONV_PART_ROW:
        convolve_rows(in, out, k, border, threads);
        break;
    case CONV_PART_COLUMN:
        convolve_columns(in, out, k, border, threads);
        break;
    case CONV_PART_TILE:
        convolve_tiles(in, out, k, border, tile_w, tile_h, threads);
        break;
    }

    return CONV_OK;
}
