#include <stdio.h>

#include <convolution/kernels.h>

#include "test_common.h"

static void test_create_shape_and_defaults(void) {
    conv_kernel *kernel = conv_kernel_create(3, 3);
    CHECK(kernel != NULL);
    if (kernel != NULL) {
        CHECK(kernel->width == 3);
        CHECK(kernel->height == 3);
        CHECK(kernel->factor == 1.0);
        CHECK(kernel->bias == 0.0);
        for (int i = 0; i < 9; ++i) {
            CHECK(kernel->data[i] == 0.0);
        }
        conv_kernel_destroy(kernel);
    }
}

static void test_create_invalid_dimensions(void) {
    CHECK(conv_kernel_create(0, 3) == NULL);
    CHECK(conv_kernel_create(3, 0) == NULL);
    CHECK(conv_kernel_create(0, 0) == NULL);
    CHECK(conv_kernel_create(-1, 3) == NULL);
    CHECK(conv_kernel_create(3, -1) == NULL);
    CHECK(conv_kernel_create(2, 3) == NULL);
    CHECK(conv_kernel_create(4, 4) == NULL);
}

static void test_validate_odd_sizes(void) {
    double coefficients[1] = {1.0};
    const int sizes[5][2] = {{1,1},{3,3},{5,5},{7,7},{3,5}};

    for (int i = 0; i < 5; ++i) {
        conv_kernel k = {sizes[i][0], sizes[i][1], coefficients, 1.0, 0.0};
        CHECK(conv_kernel_validate(&k) == CONV_OK);
    }
}

static void test_validate_even_sizes(void) {
    double coefficients[1] = {1.0};
    const int sizes[4][2] = {{2,2},{2,3},{3,2},{4,4}};

    for (int i = 0; i < 4; ++i) {
        conv_kernel k = {sizes[i][0], sizes[i][1], coefficients, 1.0, 0.0};
        CHECK(conv_kernel_validate(&k) == CONV_ERR_KERNEL_SIZE);
    }
}

static void test_validate_null_and_degenerate(void) {
    double coefficients[1] = {1.0};
    conv_kernel no_coefficients = {3, 3, NULL, 1.0, 0.0};
    conv_kernel zero_width = {0, 3, coefficients, 1.0, 0.0};
    conv_kernel zero_height = {3, 0, coefficients, 1.0, 0.0};
    conv_kernel negative_width = {-3, 3, coefficients, 1.0, 0.0};

    CHECK(conv_kernel_validate(NULL) == CONV_ERR_NULL_ARG);
    CHECK(conv_kernel_validate(&no_coefficients) == CONV_ERR_NULL_ARG);
    CHECK(conv_kernel_validate(&zero_width) == CONV_ERR_INVALID_SIZE);
    CHECK(conv_kernel_validate(&zero_height) == CONV_ERR_INVALID_SIZE);
    CHECK(conv_kernel_validate(&negative_width) == CONV_ERR_INVALID_SIZE);
}

static void test_destroy_null_is_safe(void) {
    conv_kernel_destroy(NULL);
    CHECK(1);
}

int main(void) {
    test_create_shape_and_defaults();
    test_create_invalid_dimensions();
    test_validate_odd_sizes();
    test_validate_even_sizes();
    test_validate_null_and_degenerate();
    test_destroy_null_is_safe();
    TEST_REPORT("test_kernel");
}
