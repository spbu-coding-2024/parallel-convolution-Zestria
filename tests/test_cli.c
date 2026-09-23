#include <convolution/image.h>
#include <convolution/convolution.h>

#include <stdio.h>
#include <string.h>

#include "io.h"
#include "test_common.h"

#define TMP_PNG "tmp.png"

static void test_load_null_arguments(void)
{
    conv_image *img = NULL;
    CHECK(conv_io_load_png(NULL, &img) == CONV_ERR_NULL_ARG);
    CHECK(img == NULL);
    CHECK(conv_io_load_png("unused.png", NULL) == CONV_ERR_NULL_ARG);
}

static void test_load_missing_file(void)
{
    conv_image *img = NULL;
    conv_status st = conv_io_load_png("assets/does_not_exist.png", &img);
    CHECK(st != CONV_OK);
    CHECK(img == NULL);
}

static void test_load_asset(const char *path)
{
    conv_image *img = NULL;
    conv_status st = conv_io_load_png(path, &img);
    CHECK(st == CONV_OK);
    CHECK(img != NULL);
    if (img != NULL) {
        CHECK(img->width > 0);
        CHECK(img->height > 0);
        CHECK(img->data != NULL);
        conv_image_destroy(img);
    }
}

static void test_save_null_arguments(void)
{
    CHECK(conv_io_save_png(NULL, NULL) != CONV_OK);

    conv_image *img = conv_image_create(2, 2);
    CHECK(img != NULL);
    CHECK(conv_io_save_png(NULL, img) != CONV_OK);
    CHECK(conv_io_save_png(TMP_PNG, NULL) != CONV_OK);
    conv_image_destroy(img);
}

static void test_roundtrip_preserves_pixels(void)
{
    const int W = 8;
    const int H = 8;

    conv_image *img = conv_image_create(W, H);
    CHECK(img != NULL);
    if (img == NULL) return;

    for (int i = 0; i < W * H; ++i) {
        img->data[i] = (uint8_t)(i * 4);
    }

    conv_status st_save = conv_io_save_png(TMP_PNG, img);
    CHECK(st_save == CONV_OK);

    conv_image *loaded = NULL;
    conv_status st_load = conv_io_load_png(TMP_PNG, &loaded);
    CHECK(st_load == CONV_OK);
    CHECK(loaded != NULL);

    if (loaded != NULL) {
        CHECK(loaded->width == W);
        CHECK(loaded->height == H);
        if (loaded->width == W && loaded->height == H && loaded->data != NULL) {
            for (int i = 0; i < W * H; ++i) {
                CHECK(loaded->data[i] == img->data[i]);
            }
        }
        conv_image_destroy(loaded);
    }

    conv_image_destroy(img);
    remove(TMP_PNG);
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <asset.png>\n", argv[0]);
        return 1;
    }

    test_load_null_arguments();
    test_load_missing_file();
    test_load_asset(argv[1]);
    test_save_null_arguments();
    test_roundtrip_preserves_pixels();
    TEST_REPORT("test_cli");
}

