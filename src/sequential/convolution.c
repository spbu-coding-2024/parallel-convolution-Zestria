#include <stddef.h>

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

static conv_status validate(conv_image *in, conv_image *out, conv_kernel *k) {
    if (in == NULL || k == NULL || out == NULL || in->data == NULL || out->data == NULL || k -> data == NULL) {
        return CONV_ERR_NULL_ARG;
    }
    if (in->width <= 0 || in->height <= 0 || out->width != in->width || out->height != in->height) {
        return CONV_ERR_INVALID_SIZE;
    }
    if (k->width <= 0 || k->height <= 0 || k->width % 2 == 0 || k->height % 2 == 0 || k->width > in->width || k->height > in->height) {
        return CONV_ERR_KERNEL_SIZE;
    }
    if (in->data == out->data) {
        return CONV_ERR_INVALID_PARAM;
    }
    return CONV_OK;
}

conv_status conv_apply_gray(conv_image *in, conv_image *out, conv_kernel *k) {
    conv_status st = validate(in, out, k);
    if (st != CONV_OK) {
        return st;
    }

    int w = in->width;
    int h = in->height;
    int kw = k->width;
    int kh = k->height;
    int dx = kw / 2;
    int dy = kh / 2;
    double factor = k->factor;
    double bias = k->bias;

    for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
        double acc = 0.0;

        for (int ky = 0; ky < kh; ++ky) {
        for (int kx = 0; kx < kw; ++kx) {
            int iy = (y - dy + ky + h) % h;
            int ix = (x - dx + kx + w) % w;

            acc += in->data[iy * w + ix] * k->data[ky * kw + kx];
        }
        }

        out->data[y * w + x] = clamp_u8(factor * acc + bias);
    }
    }

    return CONV_OK;
}

