#include <stdio.h>
#include <string.h>

#include <convolution/convolution.h>
#include <convolution/kernels.h>

#include "../test_common.h"
#include "../test_data.h"

static void test_null_arguments(void) {
    uint8_t in_data[9] = {0};
    uint8_t out_data[9] = {0};
    conv_image in = {3, 3, in_data};
    conv_image out = {3, 3, out_data};
    conv_image no_data = {3, 3, NULL};
    conv_kernel no_coeff = {3, 3, NULL, 1.0, 0.0};

    CHECK(conv_apply_gray(NULL, &out, &KERNEL_IDENTITY_3x3) == CONV_ERR_NULL_ARG);
    CHECK(conv_apply_gray(&in, NULL, &KERNEL_IDENTITY_3x3) == CONV_ERR_NULL_ARG);
    CHECK(conv_apply_gray(&in, &out, NULL) == CONV_ERR_NULL_ARG);
    CHECK(conv_apply_gray(&no_data, &out, &KERNEL_IDENTITY_3x3) == CONV_ERR_NULL_ARG);
    CHECK(conv_apply_gray(&in, &out, &no_coeff) == CONV_ERR_NULL_ARG);
}

static void test_invalid_geometry(void) {
    uint8_t in_data[9] = {0};
    uint8_t out_data[9] = {0};
    conv_image in = {3, 3, in_data};
    conv_image narrow = {2, 3, out_data};
    conv_image flat = {3, 2, out_data};
    conv_image zero_width = {0, 3, out_data};
    conv_image negative_height = {3, -1, out_data};

    CHECK(conv_apply_gray(&in, &narrow, &KERNEL_IDENTITY_3x3) == CONV_ERR_INVALID_SIZE);
    CHECK(conv_apply_gray(&in, &flat, &KERNEL_IDENTITY_3x3) == CONV_ERR_INVALID_SIZE);
    CHECK(conv_apply_gray(&zero_width, &narrow, &KERNEL_IDENTITY_3x3) == CONV_ERR_INVALID_SIZE);
    CHECK(conv_apply_gray(&negative_height, &flat, &KERNEL_IDENTITY_3x3) == CONV_ERR_INVALID_SIZE);
}

static void test_in_place_rejected(void) {
    uint8_t data[9] = {1,2,3,4,5,6,7,8,9};
    conv_image img = {3, 3, data};
    conv_image alias = {3, 3, data};

    CHECK(conv_apply_gray(&img, &img, &KERNEL_IDENTITY_3x3) == CONV_ERR_INVALID_PARAM);
    CHECK(conv_apply_gray(&img, &alias, &KERNEL_IDENTITY_3x3) == CONV_ERR_INVALID_PARAM);
}

static void test_kernel_dimensions(void) {
    uint8_t in_data[9] = {0};
    uint8_t out_data[9] = {0};
    conv_image in = {3, 3, in_data};
    conv_image out = {3, 3, out_data};
    double coefficients[9] = {0};
    conv_kernel even_square = {2, 2, coefficients, 1.0, 0.0};
    conv_kernel even_height = {3, 2, coefficients, 1.0, 0.0};
    conv_kernel zero_width = {0, 3, coefficients, 1.0, 0.0};
    conv_kernel negative_height = {3, -1, coefficients, 1.0, 0.0};

    CHECK(conv_apply_gray(&in, &out, &even_square) == CONV_ERR_KERNEL_SIZE);
    CHECK(conv_apply_gray(&in, &out, &even_height) == CONV_ERR_KERNEL_SIZE);
    CHECK(conv_apply_gray(&in, &out, &zero_width) == CONV_ERR_INVALID_SIZE);
    CHECK(conv_apply_gray(&in, &out, &negative_height) == CONV_ERR_INVALID_SIZE);
}

static void test_kernel_larger_than_image(void) {
    uint8_t in_data[9] = {0};
    uint8_t out_data[9] = {0};
    conv_image in = {3, 3, in_data};
    conv_image out = {3, 3, out_data};
    double coefficients[25] = {0};
    conv_kernel k = {5, 5, coefficients, 1.0, 0.0};

    CHECK(conv_apply_gray(&in, &out, &k) == CONV_ERR_KERNEL_SIZE);
}

static void test_even_kernel_rejected(void) {
    uint8_t in_data[9] = {1,2,3,4,5,6,7,8,9};
    uint8_t out_data[9] = {0};
    conv_image in = {3, 3, in_data};
    conv_image out = {3, 3, out_data};
    double coefficients[6] = {0};
    conv_kernel k = {3, 2, coefficients, 1.0, 0.0};

    CHECK(conv_apply_gray(&in, &out, &k) == CONV_ERR_KERNEL_SIZE);
}

static void test_identity_preserves_pixels(void) {
    uint8_t in3[9] = {1,2,3,4,5,6,7,8,9};
    uint8_t out3[9] = {0};
    conv_image in = {3, 3, in3};
    conv_image out = {3, 3, out3};

    CHECK(conv_apply_gray(&in, &out, &KERNEL_IDENTITY_3x3) == CONV_OK);
    CHECK(memcmp(in3, out3, 9) == 0);

    /* Non-square 4x3 image: width and height must not be mixed up. */
    uint8_t in43[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    uint8_t out43[12] = {0};
    conv_image in_ns = {4, 3, in43};
    conv_image out_ns = {4, 3, out43};

    CHECK(conv_apply_gray(&in_ns, &out_ns, &KERNEL_IDENTITY_3x3) == CONV_OK);
    CHECK(memcmp(in43, out43, 12) == 0);

    /* 1x1 image with a 1x1 identity kernel. */
    uint8_t in1[1] = {42};
    uint8_t out1[1] = {0};
    double one[1] = {1.0};
    conv_kernel k1 = {1, 1, one, 1.0, 0.0};
    conv_image in_1 = {1, 1, in1};
    conv_image out_1 = {1, 1, out1};

    CHECK(conv_apply_gray(&in_1, &out_1, &k1) == CONV_OK);
    CHECK(out1[0] == 42);
}

static void test_zero_kernel(void) {
    uint8_t in_data[9] = {1,2,3,4,5,6,7,8,9};
    uint8_t out_data[9] = {0};
    conv_image in = {3, 3, in_data};
    conv_image out = {3, 3, out_data};
    double zeros[9] = {0};
    conv_kernel k_zero = {3, 3, zeros, 1.0, 0.0};
    conv_kernel k_bias = {3, 3, zeros, 1.0, 40.0};

    CHECK(conv_apply_gray(&in, &out, &k_zero) == CONV_OK);
    for (int i = 0; i < 9; ++i) {
        CHECK(out_data[i] == 0);
    }

    uint8_t out_bias[9] = {0};
    conv_image out_b = {3, 3, out_bias};
    CHECK(conv_apply_gray(&in, &out_b, &k_bias) == CONV_OK);
    for (int i = 0; i < 9; ++i) {
        CHECK(out_bias[i] == 40);
    }
}

static void test_factor_and_bias(void) {
    double center_tap[9] = {0,0,0, 0,1,0, 0,0,0};

    /* Saturating: 2 * 200 + 100 = 500 -> 255. */
    uint8_t bright[9] = {200,200,200, 200,200,200, 200,200,200};
    uint8_t out_sat[9] = {0};
    conv_image in_bright = {3, 3, bright};
    conv_image out_b = {3, 3, out_sat};
    conv_kernel k_sat = {3, 3, center_tap, 2.0, 100.0};

    CHECK(conv_apply_gray(&in_bright, &out_b, &k_sat) == CONV_OK);
    CHECK(out_sat[0] == 255);
    CHECK(out_sat[4] == 255);
    CHECK(out_sat[8] == 255);

    /* Exact, no clamping: 2 * 10 + 0 = 20. */
    uint8_t dim[9] = {10,10,10, 10,10,10, 10,10,10};
    uint8_t out_exact[9] = {0};
    conv_image in_dim = {3, 3, dim};
    conv_image out_e = {3, 3, out_exact};
    conv_kernel k_exact = {3, 3, center_tap, 2.0, 0.0};

    CHECK(conv_apply_gray(&in_dim, &out_e, &k_exact) == CONV_OK);
    CHECK(out_exact[4] == 20);

    /* Clamped to 0: 1 * 10 - 50 = -40 -> 0. */
    uint8_t out_neg[9] = {0};
    conv_image out_n = {3, 3, out_neg};
    conv_kernel k_neg = {3, 3, center_tap, 1.0, -50.0};

    CHECK(conv_apply_gray(&in_dim, &out_n, &k_neg) == CONV_OK);
    CHECK(out_neg[4] == 0);
}

static void test_border_modes_single_tap(void) {
    uint8_t in_data[9] = {1,2,3,4,5,6,7,8,9};
    uint8_t out_data[9] = {0};
    conv_image in = {3, 3, in_data};
    conv_image out = {3, 3, out_data};
    double top_left_tap[9] = {1,0,0, 0,0,0, 0,0,0};
    conv_kernel k = {3, 3, top_left_tap, 1.0, 0.0};

    /* Output (0,0) samples (-1,-1): WRAP -> 9, CLAMP -> 1, ZERO -> 0, MIRROR -> 5. */
    out_data[0] = 0xFF;
    CHECK(conv_apply_gray_border(&in, &out, &k, CONV_BORDER_WRAP) == CONV_OK);
    CHECK(out_data[0] == 9);

    out_data[0] = 0xFF;
    CHECK(conv_apply_gray_border(&in, &out, &k, CONV_BORDER_CLAMP) == CONV_OK);
    CHECK(out_data[0] == 1);

    out_data[0] = 0xFF;
    CHECK(conv_apply_gray_border(&in, &out, &k, CONV_BORDER_ZERO) == CONV_OK);
    CHECK(out_data[0] == 0);

    out_data[0] = 0xFF;
    CHECK(conv_apply_gray_border(&in, &out, &k, CONV_BORDER_MIRROR) == CONV_OK);
    CHECK(out_data[0] == 5);
}

static void test_conv_apply_gray_matches_wrap(void) {
    const int W = 5;
    const int H = 4;
    double box[9] = {1,1,1, 1,1,1, 1,1,1};

    conv_image *in = conv_image_create(W, H);
    conv_image *default_entry = conv_image_create(W, H);
    conv_image *explicit_wrap = conv_image_create(W, H);
    CHECK(in != NULL && default_entry != NULL && explicit_wrap != NULL);
    if (in != NULL && default_entry != NULL && explicit_wrap != NULL) {
        fill_gradient(in);
        conv_kernel k = {3, 3, box, 1.0 / 9.0, 0.0};
        CHECK(conv_apply_gray(in, default_entry, &k) == CONV_OK);
        CHECK(conv_apply_gray_border(in, explicit_wrap, &k, CONV_BORDER_WRAP) == CONV_OK);
        CHECK(memcmp(default_entry->data, explicit_wrap->data, (size_t)W * (size_t)H) == 0);
    }
    conv_image_destroy(in);
    conv_image_destroy(default_entry);
    conv_image_destroy(explicit_wrap);
}

static void test_small_images(void) {
    const conv_border modes[4] = {
        CONV_BORDER_WRAP, CONV_BORDER_CLAMP, CONV_BORDER_ZERO, CONV_BORDER_MIRROR
    };
    double tap[1] = {1.0};

    for (int m = 0; m < 4; ++m) {
        uint8_t in_data[1] = {137};
        uint8_t out_data[1] = {0};
        conv_image in = {1, 1, in_data};
        conv_image out = {1, 1, out_data};
        conv_kernel k = {1, 1, tap, 1.0, 0.0};

        CHECK(conv_apply_gray_border(&in, &out, &k, modes[m]) == CONV_OK);
        CHECK(out_data[0] == 137);
    }
}

static void test_degenerate_sizes(void) {
    const conv_border modes[4] = {
        CONV_BORDER_WRAP, CONV_BORDER_CLAMP, CONV_BORDER_ZERO, CONV_BORDER_MIRROR
    };
    double box[9] = {1,1,1, 1,1,1, 1,1,1};
    conv_kernel k = {3, 3, box, 1.0 / 9.0, 0.0};

    uint8_t in_1x5[5] = {1,2,3,4,5};
    uint8_t out_1x5[5] = {0};
    conv_image in_thin_w = {1, 5, in_1x5};
    conv_image out_thin_w = {1, 5, out_1x5};

    uint8_t in_5x1[5] = {1,2,3,4,5};
    uint8_t out_5x1[5] = {0};
    conv_image in_thin_h = {5, 1, in_5x1};
    conv_image out_thin_h = {5, 1, out_5x1};

    uint8_t in_2x2[4] = {1,2,3,4};
    uint8_t out_2x2[4] = {0};
    conv_image in_small = {2, 2, in_2x2};
    conv_image out_small = {2, 2, out_2x2};

    for (int m = 0; m < 4; ++m) {
        CHECK(conv_apply_gray_border(&in_thin_w, &out_thin_w, &k, modes[m]) == CONV_ERR_KERNEL_SIZE);
        CHECK(conv_apply_gray_border(&in_thin_h, &out_thin_h, &k, modes[m]) == CONV_ERR_KERNEL_SIZE);
        CHECK(conv_apply_gray_border(&in_small, &out_small, &k, modes[m]) == CONV_ERR_KERNEL_SIZE);
    }

    /* Rejected calls must leave the output buffers untouched. */
    for (int i = 0; i < 5; ++i) {
        CHECK(out_1x5[i] == 0);
        CHECK(out_5x1[i] == 0);
    }
    for (int i = 0; i < 4; ++i) {
        CHECK(out_2x2[i] == 0);
    }
}

int main(void) {
    test_null_arguments();
    test_invalid_geometry();
    test_in_place_rejected();
    test_kernel_dimensions();
    test_kernel_larger_than_image();
    test_even_kernel_rejected();
    test_identity_preserves_pixels();
    test_zero_kernel();
    test_factor_and_bias();
    test_border_modes_single_tap();
    test_conv_apply_gray_matches_wrap();
    test_small_images();
    test_degenerate_sizes();
    TEST_REPORT("test_sequential_convolution");
}
