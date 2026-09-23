#include <stdio.h>

#include <convolution/convolution.h>
#include <convolution/image.h>
#include <convolution/kernels.h>

#include "io.h"

static void print_usage(const char *program) {
    fprintf(stderr, "usage: %s <input.png> <output.png>\n", program);
}

/* Load a PNG, convolve it with the identity kernel, and save the result. */
static int run(const char *input_path, const char *output_path) {
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

    st = conv_apply_gray(input, output, &KERNEL_IDENTITY_3x3);
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
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }
    return run(argv[1], argv[2]);
}
