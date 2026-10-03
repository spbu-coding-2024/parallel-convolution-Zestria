#include <stdio.h>
#include <string.h>

#include <convolution/convolution.h>
#include <convolution/kernels.h>
#include <convolution/pipeline.h>

#include "../test_common.h"

#define W 9
#define H 7
#define COUNT 6
#define PIXELS (W * H)

typedef struct {
    uint8_t outputs[COUNT][PIXELS];
    const conv_kernel *kernel;
    conv_border border;
    int read_fail;
    int write_fail;
} pipe_ctx;

static void fill_index(conv_image *image, int index) {
    for (int y = 0; y < image->height; ++y) {
        for (int x = 0; x < image->width; ++x) {
            image->data[y * image->width + x] =
                (uint8_t)((x * 7 + y * 13 + index * 31 + 3) % 256);
        }
    }
}

static conv_status read_image(void *ctx, int index, conv_image *in) {
    pipe_ctx *c = ctx;
    if (index == c->read_fail) {
        return CONV_ERR_IO_READ;
    }
    fill_index(in, index);
    return CONV_OK;
}

static conv_status write_image(void *ctx, int index, const conv_image *out) {
    pipe_ctx *c = ctx;
    if (index == c->write_fail) {
        return CONV_ERR_IO_WRITE;
    }
    memcpy(c->outputs[index], out->data, PIXELS);
    return CONV_OK;
}

static void check_index(const pipe_ctx *c, int index) {
    uint8_t src[PIXELS];
    uint8_t oracle[PIXELS];
    conv_image in = { W, H, src };
    conv_image out = { W, H, oracle };

    fill_index(&in, index);
    CHECK(conv_apply_gray_border(&in, &out, c->kernel, c->border) == CONV_OK);
    CHECK(memcmp(oracle, c->outputs[index], PIXELS) == 0);
}

static void run_and_check(int window, int threads, const conv_parallel_opts *par) {
    pipe_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.kernel = &KERNEL_BLUR_5x5;
    ctx.border = CONV_BORDER_WRAP;
    ctx.read_fail = -1;
    ctx.write_fail = -1;

    conv_pipeline_opts opts = { W, H, window, threads, ctx.kernel, ctx.border, par };
    CHECK(conv_pipeline_run(COUNT, &opts, read_image, write_image, &ctx) == CONV_OK);
    for (int i = 0; i < COUNT; ++i) {
        check_index(&ctx, i);
    }
}

static void test_pipeline_matches(void) {
    const int windows[] = { 1, 3 };
    const int threads[] = { 1, 4 };
    conv_parallel_opts par = { 2, CONV_PART_TILE, CONV_SCHED_STATIC, 0, 4, 4 };

    for (size_t wi = 0; wi < sizeof(windows) / sizeof(windows[0]); ++wi) {
    for (size_t ti = 0; ti < sizeof(threads) / sizeof(threads[0]); ++ti) {
        run_and_check(windows[wi], threads[ti], NULL);
        run_and_check(windows[wi], threads[ti], NULL); /* repeated run reproduces output */
    }
    }

    run_and_check(3, 4, &par);
}

static pipe_ctx plain_ctx(void) {
    pipe_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.kernel = &KERNEL_BLUR_5x5;
    ctx.border = CONV_BORDER_WRAP;
    ctx.read_fail = -1;
    ctx.write_fail = -1;
    return ctx;
}

static void test_pipeline_failures(void) {
    conv_pipeline_opts opts = { W, H, 3, 4, &KERNEL_BLUR_5x5, CONV_BORDER_WRAP, NULL };

    pipe_ctx ctx = plain_ctx();
    ctx.read_fail = 2;
    CHECK(conv_pipeline_run(COUNT, &opts, read_image, write_image, &ctx) == CONV_ERR_IO_READ);

    ctx = plain_ctx();
    ctx.write_fail = 3;
    CHECK(conv_pipeline_run(COUNT, &opts, read_image, write_image, &ctx) == CONV_ERR_IO_WRITE);
}

static void test_pipeline_invalid(void) {
    pipe_ctx ctx = plain_ctx();
    conv_pipeline_opts opts = { W, H, 3, 4, &KERNEL_BLUR_5x5, CONV_BORDER_WRAP, NULL };

    CHECK(conv_pipeline_run(COUNT, NULL, read_image, write_image, &ctx) == CONV_ERR_NULL_ARG);
    CHECK(conv_pipeline_run(COUNT, &opts, NULL, write_image, &ctx) == CONV_ERR_NULL_ARG);
    CHECK(conv_pipeline_run(COUNT, &opts, read_image, NULL, &ctx) == CONV_ERR_NULL_ARG);

    conv_pipeline_opts no_kernel = { W, H, 3, 4, NULL, CONV_BORDER_WRAP, NULL };
    CHECK(conv_pipeline_run(COUNT, &no_kernel, read_image, write_image, &ctx) == CONV_ERR_NULL_ARG);

    CHECK(conv_pipeline_run(-1, &opts, read_image, write_image, &ctx) == CONV_ERR_INVALID_PARAM);

    conv_pipeline_opts bad_window = { W, H, 0, 4, &KERNEL_BLUR_5x5, CONV_BORDER_WRAP, NULL };
    CHECK(conv_pipeline_run(COUNT, &bad_window, read_image, write_image, &ctx) == CONV_ERR_INVALID_PARAM);

    CHECK(conv_pipeline_run(0, &opts, read_image, write_image, &ctx) == CONV_OK);
}

int main(void) {
    test_pipeline_matches();
    test_pipeline_failures();
    test_pipeline_invalid();
    TEST_REPORT("test_pipeline");
}
