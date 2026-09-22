#include <stdio.h>

#include <convolution/image.h>

#include "test_common.h"

static void test_create_valid(void)
{
    conv_image *img = conv_image_create(4, 3);
    CHECK(img != NULL);
    if (img != NULL) {
        CHECK(img->width == 4);
        CHECK(img->height == 3);
        CHECK(img->data != NULL);
        for (int i = 0; i < 12; ++i) {
            CHECK(img->data[i] == 0);
        }
        conv_image_destroy(img);
    }
}

static void test_create_invalid_dimensions(void)
{
    CHECK(conv_image_create(0, 5) == NULL);
    CHECK(conv_image_create(5, 0) == NULL);
    CHECK(conv_image_create(-1, 5) == NULL);
    CHECK(conv_image_create(5, -1) == NULL);
    CHECK(conv_image_create(0, 0) == NULL);
}

static void test_destroy_null_is_safe(void)
{
    conv_image_destroy(NULL);
    CHECK(1);
}

static void test_destroy_valid(void)
{
    conv_image *img = conv_image_create(2, 2);
    CHECK(img != NULL);
    conv_image_destroy(img);
    CHECK(1);
}

int main(void)
{
    test_create_valid();
    test_create_invalid_dimensions();
    test_destroy_null_is_safe();
    test_destroy_valid();
    TEST_REPORT("test_image");
    return failures == 0 ? 0 : 1; 
}

