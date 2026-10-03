#include <stdio.h>
#include <string.h>
#include <convolution/convolution.h>
#include <convolution/kernels.h>
#include "../test_common.h"
#include "../test_data.h"
#define SIDE 9
#define PIXELS (SIDE * SIDE)

static const conv_border borders[] = {
    CONV_BORDER_WRAP, CONV_BORDER_CLAMP,
    CONV_BORDER_ZERO, CONV_BORDER_MIRROR
};

static conv_kernel *make_shift(int dx, int dy) {
    conv_kernel *k = conv_kernel_create(3, 3);
    if (k != NULL)
        k->data[(dy + 1) * 3 + dx + 1] = 1.0;
    return k;
}

static void test_compose_identity_shift(void) {
    uint8_t in_data[PIXELS], tmp_data[PIXELS], out_data[PIXELS];
    conv_image in = {SIDE, SIDE, in_data};
    conv_image tmp = {SIDE, SIDE, tmp_data};
    conv_image out = {SIDE, SIDE, out_data};
    conv_kernel *right = make_shift(1, 0);
    conv_kernel *left = make_shift(-1, 0);
    conv_kernel *both = conv_kernel_compose(right, left);
    
    CHECK(right != NULL && left != NULL && both != NULL);
    
    if (right != NULL && left != NULL && both != NULL) {
        fill_gradient(&in);
        CHECK(both->width == 5 && both->height == 5);
        CHECK(conv_apply_gray_border(&in, &tmp, right, CONV_BORDER_WRAP) == CONV_OK);
        CHECK(conv_apply_gray_border(&tmp, &out, left, CONV_BORDER_WRAP) == CONV_OK);
        CHECK(memcmp(in_data, out_data, PIXELS) == 0);
        CHECK(conv_apply_gray_border(&in, &out, both, CONV_BORDER_WRAP) == CONV_OK);
        CHECK(memcmp(in_data, out_data, PIXELS) == 0);
    }
    
    conv_kernel_destroy(right);
    conv_kernel_destroy(left);
    conv_kernel_destroy(both);
}

static void test_identity_and_zero(void) {
    uint8_t in_data[PIXELS], out_data[PIXELS];
    conv_image in = {SIDE, SIDE, in_data};
    conv_image out = {SIDE, SIDE, out_data};
    double zero_data[9] = {0};
    conv_kernel zero = {3, 3, zero_data, 1.0, 0.0};

    fill_gradient(&in);
    
    for (size_t b = 0; b < sizeof(borders) / sizeof(borders[0]); ++b) {
        CHECK(conv_apply_gray_border(&in, &out, &KERNEL_IDENTITY_3x3, borders[b]) == CONV_OK);
        CHECK(memcmp(in_data, out_data, PIXELS) == 0);
        CHECK(conv_apply_gray_border(&in, &out, &zero, borders[b]) == CONV_OK);
        for (int i = 0; i < PIXELS; ++i)
            CHECK(out_data[i] == 0);
    }
}

static void test_padding(void) {
    conv_kernel *wide = conv_kernel_pad(&KERNEL_BLUR_3x3, 5, 5);
    uint8_t in_data[PIXELS], a_data[PIXELS], b_data[PIXELS];
    conv_image in = {SIDE, SIDE, in_data};
    conv_image a = {SIDE, SIDE, a_data};
    conv_image b = {SIDE, SIDE, b_data};
    
    CHECK(wide != NULL);
    
    if (wide != NULL) {
        fill_gradient(&in);
        for (size_t i = 0; i < sizeof(borders) / sizeof(borders[0]); ++i) {
            CHECK(conv_apply_gray_border(&in, &a, &KERNEL_BLUR_3x3, borders[i]) == CONV_OK);
            CHECK(conv_apply_gray_border(&in, &b, wide, borders[i]) == CONV_OK);
            CHECK(memcmp(a_data, b_data, PIXELS) == 0);
        }
    }
    conv_kernel_destroy(wide);
}

int main(void) {
    test_compose_identity_shift();
    test_identity_and_zero();
    test_padding();
    TEST_REPORT("test_composition");
}
