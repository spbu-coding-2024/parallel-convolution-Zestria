#pragma once

#include "stb_image.h"
#include "stb_image_write.h"

typedef enum
{
    STRATEGY_PIXEL = 0,
    STRATEGY_ROW,
    STRATEGY_COL,
    STRATEGY_TILE
} Strategy;


typedef struct
{
    const stbi_uc *input;
    stbi_uc *output;
    int width;
    int height;
    int channels;
    const double *kernel;
    int kernel_size;
    double factor;
    double bias;
} ConvArgs;

typedef struct
{
    const char *name;
    const double *kernel;
    int kernel_size;
    double factor;
    double bias;
} Filter;

typedef struct
{
    const Filter **steps;
    int n_steps;
} Pipeline;

static const double k_identity[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
static const double k_blur3[] = {0.0, 0.2, 0.0, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0};
static const double k_blur5[] = {
    0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
};
static const double k_gaussian3[] = {
    0.077847, 0.123317, 0.077847, 0.123317, 0.195346, 0.123317, 0.077847, 0.123317, 0.077847,
};
static const double k_motion9[] = {
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
};
static const double k_find5_h[] = {
    0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
static const double k_find5_v[] = {
    0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 4, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
};

// Filter registry

#define FILTER(n, k, sz, f, b)                                                                     \
    {                                                                                              \
        .name = (n), .kernel = (k), .kernel_size = (sz), .factor = (f), .bias = (b)                \
    }

static const Filter g_registry[] = {
    FILTER("identity", k_identity, 3, 1.0, 0.0),    FILTER("blur3", k_blur3, 3, 1, 0),
    FILTER("blur5", k_blur5, 5, 1.0 / 13.0, 0.0),   FILTER("gaussian", k_gaussian3, 3, 1.0, 0.0),
    FILTER("motion", k_motion9, 9, 1.0 / 9.0, 0.0), FILTER("find5_h", k_find5_h, 5, 1.0, 0.0),
    FILTER("find5_v", k_find5_v, 5, 1.0, 0.0)};

static const int g_registry_count = (int)(sizeof g_registry / sizeof g_registry[0]);