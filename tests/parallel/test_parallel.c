#include <stdio.h>
#include <string.h>

#include <convolution/convolution.h>
#include <convolution/kernels.h>

#include "../test_common.h"
#include "../test_data.h"

#define MAX_PIXELS (13 * 7)

typedef struct {
    int width;
    int height;
} image_size;

static const image_size sizes[] = {
    {9, 9},
    {13, 7},
};

static const conv_kernel *kernels[] = { &KERNEL_IDENTITY_3x3, &KERNEL_BLUR_5x5 };

static const conv_border borders[] = {
    CONV_BORDER_WRAP, CONV_BORDER_CLAMP, CONV_BORDER_ZERO, CONV_BORDER_MIRROR
};

static const conv_partition partitions[] = {
    CONV_PART_PIXEL, CONV_PART_ROW, CONV_PART_COLUMN, CONV_PART_TILE
};

static const conv_omp_schedule schedules[] = {
    CONV_SCHED_STATIC, CONV_SCHED_DYNAMIC, CONV_SCHED_GUIDED, CONV_SCHED_RUNTIME
};

static const int thread_counts[] = { 1, 4 };

static void check_parallel(const conv_image *in, const conv_kernel *k, conv_border border,
                           const conv_parallel_opts *opts) {
    uint8_t oracle[MAX_PIXELS];
    uint8_t actual[MAX_PIXELS];
    conv_image expected = {in->width, in->height, oracle};
    conv_image out = {in->width, in->height, actual};

    CHECK(conv_apply_gray_border(in, &expected, k, border) == CONV_OK);
    CHECK(conv_apply_gray_parallel(in, &out, k, border, opts) == CONV_OK);
    CHECK(memcmp(oracle, actual, (size_t)in->width * (size_t)in->height) == 0);
}

static void test_parallel_matches(void) {
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
        uint8_t data[MAX_PIXELS];
        conv_image in = {sizes[s].width, sizes[s].height, data};
        fill_gradient(&in);

        for (size_t ki = 0; ki < sizeof(kernels) / sizeof(kernels[0]); ++ki) {
        for (size_t b = 0; b < sizeof(borders) / sizeof(borders[0]); ++b) {
        for (size_t p = 0; p < sizeof(partitions) / sizeof(partitions[0]); ++p) {
        for (size_t t = 0; t < sizeof(thread_counts) / sizeof(thread_counts[0]); ++t) {
        for (size_t sc = 0; sc < sizeof(schedules) / sizeof(schedules[0]); ++sc) {
            conv_parallel_opts opts = {
                thread_counts[t], partitions[p], schedules[sc], 1, 4, 4
            };
            check_parallel(&in, kernels[ki], borders[b], &opts);
        }
        }
        }
        }
        }

        conv_parallel_opts zeros = { 0, CONV_PART_TILE, CONV_SCHED_STATIC, 0, 0, 0 };
        check_parallel(&in, &KERNEL_IDENTITY_3x3, CONV_BORDER_WRAP, NULL);
        check_parallel(&in, &KERNEL_IDENTITY_3x3, CONV_BORDER_WRAP, &zeros);
    }
}

int main(void) {
    test_parallel_matches();
    TEST_REPORT("test_parallel");
}
