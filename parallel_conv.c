#ifndef CONV_TEST_BUILD

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#endif

#include "conv_common.h"

#include <strings.h>

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Registry helpers

static const Filter *filter_find(const char *name)
{
    for (int i = 0; i < g_registry_count; ++i)
        if (strcmp(g_registry[i].name, name) == 0)
            return &g_registry[i];
    return NULL;
}

static void filter_list(void)
{
    fprintf(stderr, "Available filters:\n");
    for (int i = 0; i < g_registry_count; ++i)
        fprintf(stderr, "  %-20s (%dx%d kernel)\n", g_registry[i].name, g_registry[i].kernel_size,
                g_registry[i].kernel_size);
}

static Pipeline pipeline_create(void)
{
    return (Pipeline) {NULL, 0};
}

// Pipeline helpers
static int pipeline_add(Pipeline *p, const char *name)
{
    const Filter *f = filter_find(name);
    if (!f)
    {
        fprintf(stderr, "Unknown filter: '%s'\n", name);
        filter_list();
        return -1;
    }
    p->steps = realloc(p->steps, (size_t)(p->n_steps + 1) * sizeof *p->steps);
    p->steps[p->n_steps++] = f;
    return 0;
}

static void pipeline_free(Pipeline *p)
{
    free(p->steps);
    p->steps = NULL;
    p->n_steps = 0;
}

// Convolution helpers
static inline stbi_uc clamp_to_byte(double v, double factor, double bias)
{
    double r = factor * v + bias;
    if (r < 0.0)
        return 0;
    if (r > 255.0)
        return 255;
    return (stbi_uc)round(r);
}

static inline void compute_pixel(const ConvArgs *a, int ox, int oy, int c)
{
    const int half = a->kernel_size / 2;
    double acc = 0.0;
    for (int ky = 0; ky < a->kernel_size; ++ky)
    {
        for (int kx = 0; kx < a->kernel_size; ++kx)
        {
            int ix = ox - half + kx;
            int iy = oy - half + ky;

            ix %= a->width;
            if (ix < 0)
                ix += a->width;
            iy %= a->height;
            if (iy < 0)
                iy += a->height;
            acc += a->kernel[ky * a->kernel_size + kx] *
                   (double)a->input[(iy * a->width + ix) * a->channels + c];
        }
    }
    a->output[(oy * a->width + ox) * a->channels + c] = clamp_to_byte(acc, a->factor, a->bias);
}

// Strategy implementations

void convolve_pixel(ConvArgs *a)
{
    const long total = (long)a->width * a->height;

#pragma omp parallel for
    for (long idx = 0; idx < total; ++idx)
    {
        int ox = (int)(idx % a->width);
        int oy = (int)(idx / a->width);
        for (int c = 0; c < a->channels; ++c)
            compute_pixel(a, ox, oy, c);
    }
}

void convolve_row(ConvArgs *a)
{
#pragma omp parallel for
    for (int oy = 0; oy < a->height; ++oy)
    {
        for (int ox = 0; ox < a->width; ++ox)
        {
            for (int c = 0; c < a->channels; ++c)
                compute_pixel(a, ox, oy, c);
        }
    }
}

void convolve_col(ConvArgs *a)
{
#pragma omp parallel for
    for (int ox = 0; ox < a->width; ++ox)
    {
        for (int oy = 0; oy < a->height; ++oy)
        {
            for (int c = 0; c < a->channels; ++c)
                compute_pixel(a, ox, oy, c);
        }
    }
}

void convolve_tile(ConvArgs *a, int tile_w, int tile_h)
{
    int ntx = (a->width + tile_w - 1) / tile_w;
    int nty = (a->height + tile_h - 1) / tile_h;
    int total_tiles = ntx * nty;

#pragma omp parallel for
    for (int t = 0; t < total_tiles; ++t)
    {
        int tx = t % ntx;
        int ty = t / ntx;

        int x0 = tx * tile_w, x1 = x0 + tile_w < a->width ? x0 + tile_w : a->width;
        int y0 = ty * tile_h, y1 = y0 + tile_h < a->height ? y0 + tile_h : a->height;

        for (int oy = y0; oy < y1; ++oy)
        {
            for (int ox = x0; ox < x1; ++ox)
            {
                for (int c = 0; c < a->channels; ++c)
                    compute_pixel(a, ox, oy, c);
            }
        }
    }
}

static void run_strategy(ConvArgs *a, Strategy strategy, int tile_w, int tile_h)
{
    switch (strategy)
    {
    case STRATEGY_PIXEL :
        convolve_pixel(a);
        break;
    case STRATEGY_ROW :
        convolve_row(a);
        break;
    case STRATEGY_COL :
        convolve_col(a);
        break;
    case STRATEGY_TILE :
        convolve_tile(a, tile_w, tile_h);
        break;
    }
}

static int pipeline_run(const Pipeline *p, const stbi_uc *input, stbi_uc *output, int width,
                        int height, int channels, Strategy strategy, int tile_w, int tile_h)
{

    if (p->n_steps == 0)
    {
        memcpy(output, input, (size_t)(width * height * channels));
        return 0;
    }

    const size_t img_bytes = (size_t)(width * height * channels);

    stbi_uc *buf[2] = {NULL, NULL};
    if (p->n_steps > 1)
    {
        buf[0] = malloc(img_bytes);
        buf[1] = malloc(img_bytes);
        if (!buf[0] || !buf[1])
        {
            fprintf(stderr, "Error: out of memory for pipeline buffers\n");
            free(buf[0]);
            free(buf[1]);
            return -1;
        }
    }

    for (int s = 0; s < p->n_steps; ++s)
    {
        const Filter *f = p->steps[s];

        const stbi_uc *src = (s == 0) ? input : buf[(s - 1) % 2];

        stbi_uc *dst;
        if (s == p->n_steps - 1)
            dst = output;
        else
            dst = buf[s % 2];

        ConvArgs args = {
            .input = src,
            .output = dst,
            .width = width,
            .height = height,
            .channels = channels,
            .kernel = f->kernel,
            .kernel_size = f->kernel_size,
            .factor = f->factor,
            .bias = f->bias,
        };

        run_strategy(&args, strategy, tile_w, tile_h);
    }

    free(buf[0]);
    free(buf[1]);
    return 0;
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#ifndef CONV_TEST_BUILD
int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        fprintf(stderr,
                "Usage: %s <input> <output> <strategy> [tile_w tile_h] --filter [filter ...]\n"
                "  strategy: pixel | row | col | tile\n"
                "  tile_w, tile_h: only used for strategy=tile (default 64 64)\n"
                "  filters: one or more filter applied in order\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];
    const char *strategy_s = argv[3];

    Strategy strategy;
    if (strcmp(strategy_s, "pixel") == 0)
        strategy = STRATEGY_PIXEL;
    else if (strcmp(strategy_s, "row") == 0)
        strategy = STRATEGY_ROW;
    else if (strcmp(strategy_s, "col") == 0)
        strategy = STRATEGY_COL;
    else if (strcmp(strategy_s, "tile") == 0)
        strategy = STRATEGY_TILE;
    else
    {
        fprintf(stderr, "Unknown strategy '%s'\n", strategy_s);
        return EXIT_FAILURE;
    }

    int tile_w = 64, tile_h = 64;
    int filter_start = 4;

    if (argc > filter_start + 1 && strcmp(argv[filter_start], "--filter") != 0)
    {
        tile_w = atoi(argv[filter_start]);
        filter_start++;
    }

    if (argc > filter_start + 1 && strcmp(argv[filter_start], "--filter") != 0)
    {
        tile_h = atoi(argv[filter_start]);
        filter_start++;
    }

    if (filter_start < argc && strcmp(argv[filter_start], "--filter") == 0)
        ++filter_start;

    Pipeline pipeline = pipeline_create();
    if (filter_start >= argc)
    {
        fprintf(stderr, "No filters specified; defaulting to 'identity'\n");
        pipeline_add(&pipeline, "identity");
    }
    else
    {
        for (int i = filter_start; i < argc; ++i)
            if (pipeline_add(&pipeline, argv[i]) != 0)
            {
                pipeline_free(&pipeline);
                return EXIT_FAILURE;
            }
    }

    // Image loading
    int width, height, orig_ch;
    stbi_uc *image = stbi_load(input_path, &width, &height, &orig_ch, 4);
    if (!image)
    {
        fprintf(stderr, "Error loading '%s': %s\n", input_path, stbi_failure_reason());
        pipeline_free(&pipeline);
        return EXIT_FAILURE;
    }
    printf("Loaded '%s': %d×%d px, %d channel(s)\n", input_path, width, height, orig_ch);

    stbi_uc *output = malloc((size_t)(width * height * 4));
    if (!output)
    {
        fprintf(stderr, "Error: out of memory\n");
        stbi_image_free(image);
        pipeline_free(&pipeline);
        return EXIT_FAILURE;
    }

    printf("Strategy: %s", strategy_s);
    if (strategy == STRATEGY_TILE)
        printf(" (tile %dx%d)", tile_w, tile_h);
    printf(" | threads: %d | steps: %d\n", omp_get_max_threads(), pipeline.n_steps);

    /* Run selected strategy, timed */
    double t0 = now_sec();

    int rc = pipeline_run(&pipeline, image, output, width, height, 4, strategy, tile_w, tile_h);

    if (rc != 0)
    {
        free(output);
        stbi_image_free(image);
        pipeline_free(&pipeline);
        return EXIT_FAILURE;
    }
    printf("Pipeline complete in %.3f s\n", now_sec() - t0);

    /* Preserve original alpha channel */
    for (int i = 0; i < width * height; ++i)
        output[i * 4 + 3] = image[i * 4 + 3];

    /* Write output */
    if (!stbi_write_png(output_path, width, height, 4, output, width * 4))
    {
        fprintf(stderr, "Error: failed to write '%s'\n", output_path);
        free(output);
        stbi_image_free(image);
        pipeline_free(&pipeline);
        return EXIT_FAILURE;
    }
    printf("Result saved to '%s'\n", output_path);

    free(output);
    stbi_image_free(image);
    pipeline_free(&pipeline);
    return EXIT_SUCCESS;
}
#endif
