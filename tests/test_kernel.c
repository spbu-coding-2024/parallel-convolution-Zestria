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
}

static void test_destroy_null_is_safe(void) {
    conv_kernel_destroy(NULL);
    CHECK(1);
}

int main(void) {
    test_create_shape_and_defaults();
    test_create_invalid_dimensions();
    test_destroy_null_is_safe();
    TEST_REPORT("test_kernel");
}
