#include <stdio.h>
#include <stdlib.h>
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

typedef struct {
    const char *name;
    conv_partition partition;
} partition_choice;

static const partition_choice PARTITIONS[] = {
    {"pixel",  CONV_PART_PIXEL},
    {"row",    CONV_PART_ROW},
    {"column", CONV_PART_COLUMN},
    {"tile",   CONV_PART_TILE}
};

typedef struct {
    const char *name;
    conv_omp_schedule schedule;
} schedule_choice;

static const schedule_choice SCHEDULES[] = {
    {"static",  CONV_SCHED_STATIC},
    {"dynamic", CONV_SCHED_DYNAMIC},
    {"guided",  CONV_SCHED_GUIDED},
    {"runtime", CONV_SCHED_RUNTIME}
};

static void print_usage(const char *program) {
    fprintf(stderr,
            "usage: %s [--kernel <name>] [--partition <name>] [--threads <n>] "
            "[--schedule <name>] <input.png> <output.png>\n",
            program);
    fprintf(stderr, "kernels:");
    for (size_t i = 0; i < KERNEL_COUNT; ++i) {
        fprintf(stderr, " %s", KERNELS[i].name);
    }
    fprintf(stderr, "\npartitions:");
    for (size_t i = 0; i < sizeof(PARTITIONS) / sizeof(PARTITIONS[0]); ++i) {
        fprintf(stderr, " %s", PARTITIONS[i].name);
    }
    fprintf(stderr, "\nschedules:");
    for (size_t i = 0; i < sizeof(SCHEDULES) / sizeof(SCHEDULES[0]); ++i) {
        fprintf(stderr, " %s", SCHEDULES[i].name);
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

static int find_partition(const char *name, conv_partition *partition) {
    for (size_t i = 0; i < sizeof(PARTITIONS) / sizeof(PARTITIONS[0]); ++i) {
        if (strcmp(PARTITIONS[i].name, name) == 0) {
            *partition = PARTITIONS[i].partition;
            return 1;
        }
    }
    return 0;
}

static int find_schedule(const char *name, conv_omp_schedule *schedule) {
    for (size_t i = 0; i < sizeof(SCHEDULES) / sizeof(SCHEDULES[0]); ++i) {
        if (strcmp(SCHEDULES[i].name, name) == 0) {
            *schedule = SCHEDULES[i].schedule;
            return 1;
        }
    }
    return 0;
}

static int run(const char *input_path, const char *output_path, const conv_kernel *kernel,
               const conv_parallel_opts *opts) {
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

    st = (opts != NULL)
        ? conv_apply_gray_parallel(input, output, kernel, CONV_BORDER_WRAP, opts)
        : conv_apply_gray(input, output, kernel);
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
    conv_parallel_opts opts = { 0, CONV_PART_PIXEL, CONV_SCHED_STATIC, 0, 0, 0 };
    int parallel = 0;
    int i = 1;

    while (i < argc && strncmp(argv[i], "--", 2) == 0) {
        if (i + 1 >= argc) {
            print_usage(argv[0]);
            return 1;
        }
        const char *value = argv[i + 1];

        if (strcmp(argv[i], "--kernel") == 0) {
            kernel = find_kernel(value);
            if (kernel == NULL) {
                fprintf(stderr, "error: unknown kernel '%s'\n", value);
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "--partition") == 0) {
            if (!find_partition(value, &opts.partition)) {
                fprintf(stderr, "error: unknown partition '%s'\n", value);
                print_usage(argv[0]);
                return 1;
            }
            parallel = 1;
        } else if (strcmp(argv[i], "--threads") == 0) {
            opts.threads = atoi(value);
        } else if (strcmp(argv[i], "--schedule") == 0) {
            if (!find_schedule(value, &opts.schedule)) {
                fprintf(stderr, "error: unknown schedule '%s'\n", value);
                print_usage(argv[0]);
                return 1;
            }
        } else {
            print_usage(argv[0]);
            return 1;
        }

        i += 2;
    }

    if (argc - i != 2) {
        print_usage(argv[0]);
        return 1;
    }
    return run(argv[i], argv[i + 1], kernel, parallel ? &opts : NULL);
}
