#include <stdio.h>
#include <string.h>

#include <convolution/convolution.h>
#include <convolution/image.h>
#include <convolution/kernels.h>

#include "io.h"

typedef struct {
    const char *name;
    const conv_kernel *kernel;
} kernel_choice;

static const kernel_choice KERNELS[] = {
    {"identity_3x3",              &KERNEL_IDENTITY_3x3},
    {"blur_3x3",                  &KERNEL_BLUR_3x3},
    {"blur_5x5",                  &KERNEL_BLUR_5x5},
    {"gaussian_blur_3x3",         &KERNEL_GAUSSIAN_BLUR_3x3},
    {"motion_blur_9x9",           &KERNEL_MOTION_BLUR_9x9},
    {"find_horizontal_edges_5x5", &KERNEL_FIND_HORIZONTAL_EDGES_5x5},
    {"find_vertical_edges_5x5",   &KERNEL_FIND_VERTICAL_EDGES_5x5},
    {"find_diagonal_edges_5x5",   &KERNEL_FIND_DIAGONAL_EDGES_5x5},
    {"find_all_edges_3x3",        &KERNEL_FIND_ALL_EDGES_3x3},
    {"sharpen_3x3",               &KERNEL_SHARPEN_3x3},
    {"sharpen_5x5",               &KERNEL_SHARPEN_5x5},
    {"sharpen_excessively_3x3",   &KERNEL_SHARPEN_EXCESSIVELY_3x3},
    {"emboss_3x3",                &KERNEL_EMBOSS_3x3},
    {"emboss_5x5",                &KERNEL_EMBOSS_5x5},
    {"mean_3x3",                  &KERNEL_MEAN_3x3}
};

#define KERNEL_COUNT (sizeof(KERNELS) / sizeof(KERNELS[0]))

static void print_usage(const char *program) {
    fprintf(stderr, "usage: %s [--kernel <name>] <input.png> <output.png>\n", program);
    fprintf(stderr, "kernels:");
    for (size_t i = 0; i < KERNEL_COUNT; ++i) {
        fprintf(stderr, " %s", KERNELS[i].name);
    }
    fprintf(stderr, "\n");
}

static const conv_kernel *find_kernel(const char *name) {
    for (size_t i = 0; i < KERNEL_COUNT; ++i) {
        if (strcmp(KERNELS[i].name, name) == 0) {
            return KERNELS[i].kernel;
        }
    }
    return NULL;
}

static int run(const char *input_path, const char *output_path, const conv_kernel *kernel) {
    conv_image *input = NULL;
    conv_status st = conv_io_load_png(input_path, &input);
    if (st != CONV_OK) {
        fprintf(stderr, "error: cannot load '%s' (status %d)\n",
                input_path, (int)st);
        return 1;
    }

    conv_image *output = conv_image_create(input->width, input->height);
    if (output == NULL) {
        fprintf(stderr, "error: cannot allocate a %dx%d output image\n",
                input->width, input->height);
        conv_image_destroy(input);
        return 1;
    }

    st = conv_apply_gray(input, output, kernel);
    if (st != CONV_OK) {
        fprintf(stderr, "error: convolution of '%s' failed (status %d)\n",
                input_path, (int)st);
        conv_image_destroy(input);
        conv_image_destroy(output);
        return 1;
    }

    st = conv_io_save_png(output_path, output);
    if (st != CONV_OK) {
        fprintf(stderr, "error: cannot save '%s' (status %d)\n",
                output_path, (int)st);
        conv_image_destroy(input);
        conv_image_destroy(output);
        return 1;
    }

    conv_image_destroy(input);
    conv_image_destroy(output);
    return 0;
}

int main(int argc, char *argv[]) {
    const conv_kernel *kernel = &KERNEL_IDENTITY_3x3;
    int first_path = 1;

    if (argc > 1 && strcmp(argv[1], "--kernel") == 0) {
        if (argc != 5) {
            print_usage(argv[0]);
            return 1;
        }
        kernel = find_kernel(argv[2]);
        if (kernel == NULL) {
            fprintf(stderr, "error: unknown kernel '%s'\n", argv[2]);
            print_usage(argv[0]);
            return 1;
        }
        first_path = 3;
    }

    if (argc != first_path + 2) {
        print_usage(argv[0]);
        return 1;
    }
    return run(argv[first_path], argv[first_path + 1], kernel);
}
