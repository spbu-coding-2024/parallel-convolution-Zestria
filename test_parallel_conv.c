
#define CONV_TEST_BUILD

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "conv_common.h"
#include "parallel_conv.c"
#include "sequential_conv.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_passed = 0;
static int g_failed = 0;
static int g_total = 0;

static void test_begin(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    printf("  [     ] ");
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
    ++g_total;
}

static void test_pass(void)
{
    printf("\r [Passed]\n");
    ++g_passed;
}

static void test_fail(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    printf("\r [Failed] ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    fflush(stdout);
    ++g_failed;
}

// Image helpers

static stbi_uc *img_alloc(int w, int h, int c, stbi_uc fill)
{
    stbi_uc *p = malloc((size_t)(w * h * c));
    assert(p);
    memset(p, fill, (size_t)(w * h * c));
    return p;
}

static unsigned xorshift32(unsigned *state)
{
    unsigned x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static stbi_uc *img_random(int w, int h, int c, unsigned seed)
{
    stbi_uc *p = malloc((size_t)(w * h * c));
    assert(p);
    unsigned s = seed ? seed : 1;
    for (int i = 0; i < w * h * c; ++i)
        p[i] = (stbi_uc)(xorshift32(&s) & 0xFF);

    return p;
}

static int img_max_diff(const stbi_uc *a, const stbi_uc *b, int w, int h, int c)
{
    int mx = 0;
    for (int i = 0; i < w * h * c; ++i)
    {
        int d = (int)a[i] - (int)b[i];
        if (d < 0)
            d = -d;
        if (d > mx)
            mx = d;
    }
    return mx;
}

// Kernel helpers

static double *kernel_zeros(int ksz)
{
    double *k = calloc((size_t)(ksz * ksz), sizeof(double));
    assert(k);
    return k;
}

static double *kernel_shift(int ksz, int dx, int dy)
{
    double *k = kernel_zeros(ksz);
    int cx = ksz / 2 + dx;
    int cy = ksz / 2 + dy;
    assert(cx >= 0 && cx < ksz && cy >= 0 && cy < ksz);
    k[cy * ksz + cx] = 1.0;
    return k;
}

static double *kernel_identity(int ksz)
{
    return kernel_shift(ksz, 0, 0);
}

static double *kernel_pad(const double *src, int src_sz, int dst_sz)
{
    assert(dst_sz >= src_sz);
    assert((dst_sz & 1) == 1 && (src_sz & 1) == 1);
    double *k = kernel_zeros(dst_sz);
    int off = (dst_sz - src_sz) / 2;
    for (int r = 0; r < src_sz; ++r)
        for (int c = 0; c < src_sz; ++c)
            k[(r + off) * dst_sz + (c + off)] = src[r * src_sz + c];
    return k;
}

// Pipeline helpers

static stbi_uc *paralell_run(const stbi_uc *input, int w, int h, int ch, const Filter **steps,
                             int n_steps, Strategy strategy)
{
    stbi_uc *out = malloc((size_t)(w * h * ch));
    assert(out);

    Pipeline p = pipeline_create();
    for (int i = 0; i < n_steps; ++i)
    {
        p.steps = realloc(p.steps, (size_t)(p.n_steps + 1) * sizeof *p.steps);
        p.steps[p.n_steps++] = steps[i];
    }
    pipeline_run(&p, input, out, w, h, ch, strategy, 8, 8);
    pipeline_free(&p);
    return out;
}

static stbi_uc *sequential_run(const stbi_uc *input, int w, int h, int ch, const Filter **steps,
                               int n_steps)
{

    if (n_steps == 0)
    {
        stbi_uc *out = malloc((size_t)(w * h * ch));
        assert(out);
        memcpy(out, input, (size_t)(w * h * ch));
        return out;
    }

    const size_t sz = (size_t)(w * h * ch);
    stbi_uc *buf_a = malloc(sz);
    stbi_uc *buf_b = malloc(sz);
    assert(buf_a && buf_b);

    const stbi_uc *src = input;
    stbi_uc *dst = buf_a;

    for (int s = 0; s < n_steps; ++s)
    {
        ConvArgs args = {
            .input = src,
            .output = dst,
            .width = w,
            .height = h,
            .channels = ch,
            .kernel = steps[s]->kernel,
            .kernel_size = steps[s]->kernel_size,
            .factor = steps[s]->factor,
            .bias = steps[s]->bias,
        };

        convolve_sequential(&args);

        if (s < n_steps - 1)
        {
            src = dst;
            dst = (dst == buf_a) ? buf_b : buf_a;
        }
    }

    free(dst == buf_a ? buf_b : buf_a);

    return dst;
}

// Test dimensions

static const int g_test_dims[][2] = {{1, 1}, {3, 3},   {5, 5},     {7, 7},     {7, 11},
                                     {8, 8}, {11, 13}, {100, 100}, {250, 125}, {0, 0}};

static const int g_test_ksz[] = {1, 3, 5, 7, 9, 0};

static void test_zero_kernel(void)
{
    for (int di = 0; g_test_dims[di][0]; ++di)
    {
        int w = g_test_dims[di][0], h = g_test_dims[di][1];
        for (int ki = 0; g_test_ksz[ki]; ++ki)
        {
            int ksz = g_test_ksz[ki];
            test_begin("zero kernel ksz=%d image %dx%d", ksz, w, h);

            double *k = kernel_zeros(ksz);
            stbi_uc *img = img_random(w, h, 1, (unsigned)(di + ki + ksz + time(NULL)));
            stbi_uc *out = img_alloc(w, h, 1, 0);

            ConvArgs a = {img, out, w, h, 1, k, ksz, 1.0, 0.0};
            convolve_row(&a);
            int ok = 1;
            for (int i = 0; i < w * h; ++i)
                if (out[i] != 0)
                {
                    ok = 0;
                    break;
                }
            if (!ok)
            {
                free(k);
                free(img);
                free(out);
                test_fail("expected all-zero output");
                continue;
            }

            a.bias = 128.0;
            convolve_row(&a);
            ok = 1;
            for (int i = 0; i < w * h; ++i)
                if (out[i] != 128)
                {
                    ok = 0;
                    break;
                }

            free(k);
            free(img);
            free(out);
            if (!ok)
            {
                test_fail("expected all-128 output for bias=128");
                continue;
            }
            test_pass();
        }
    }
}

static void test_identity_kernel(void)
{
    for (int di = 0; g_test_dims[di][0]; ++di)
    {
        int w = g_test_dims[di][0], h = g_test_dims[di][1];
        for (int ki = 0; g_test_ksz[ki]; ++ki)
        {
            int ksz = g_test_ksz[ki];
            test_begin("identity kernel ksz=%d image %dx%d", ksz, w, h);

            double *k = kernel_identity(ksz);
            stbi_uc *img = img_random(w, h, 3, (unsigned)(w * 137 + h * 17 + ksz + time(NULL)));
            stbi_uc *out = img_alloc(w, h, 3, 0);

            ConvArgs a = {img, out, w, h, 3, k, ksz, 1.0, 0.0};
            convolve_row(&a);

            int diff = img_max_diff(img, out, w, h, 3);
            free(k);
            free(img);
            free(out);

            if (diff != 0)
                test_fail("max pixel diff = %d, expected 0", diff);
            else
                test_pass();
        }
    }
}

static void test_zero_padding_invariance(void)
{
    for (int ri = 0; ri < g_registry_count; ++ri)
    {
        const Filter *f = &g_registry[ri];

        for (int extra = 2; extra <= 4; extra += 2)
        {
            int ksz2 = f->kernel_size + extra;
            test_begin("zero-pad %s (%d->%d)", f->name, f->kernel_size, ksz2);

            double *kpad = kernel_pad(f->kernel, f->kernel_size, ksz2);

            for (int di = 0; g_test_dims[di][0]; ++di)
            {
                int w = g_test_dims[di][0], h = g_test_dims[di][1];
                stbi_uc *img = img_random(w, h, 1, (unsigned)(ri + w + h + time(NULL)));
                stbi_uc *out1 = img_alloc(w, h, 1, 0);
                stbi_uc *out2 = img_alloc(w, h, 1, 0);

                ConvArgs a1 = {img, out1, w, h, 1, f->kernel, f->kernel_size, f->factor, f->bias};
                ConvArgs a2 = {img, out2, w, h, 1, kpad, ksz2, f->factor, f->bias};

                convolve_row(&a1);
                convolve_row(&a2);

                int diff = img_max_diff(out1, out2, w, h, 1);
                free(img);
                free(out1);
                free(out2);

                if (diff != 0)
                {
                    test_fail("filter '%s' pad=%d image %dx%d max diff=%d", f->name, extra, w, h,
                              diff);
                    goto next_kernel;
                }
            }
            free(kpad);
            test_pass();
        next_kernel:;
        }
    }
}

static void test_compositionality(void)
{
    // parallel_pipeline(A, B) == sequential_pipeline(A, B)
    for (int ai = 0; ai < g_registry_count; ++ai)
    {
        const Filter *fa = &g_registry[ai];
        for (int bi = 0; bi < g_registry_count; ++bi)
        {
            const Filter *fb = &g_registry[bi];

            test_begin("parallel==sequential: pipeline(%s, %s)", fa->name, fb->name);

            int ok = 1;
            for (int di = 0; g_test_dims[di][0] && ok; ++di)
            {
                int w = g_test_dims[di][0], h = g_test_dims[di][1];

                stbi_uc *img =
                    img_random(w, h, 1, (unsigned)(ai * 997 + bi * 101 + w * 13 + h + time(NULL)));

                const Filter *steps[2] = {fa, fb};

                stbi_uc *seq_out = sequential_run(img, w, h, 1, steps, 2);
                stbi_uc *parallel_out = paralell_run(img, w, h, 1, steps, 2, STRATEGY_ROW);
                int diff = img_max_diff(seq_out, parallel_out, w, h, 1);
                free(img);
                free(seq_out);
                free(parallel_out);

                if (diff != 0)
                {
                    test_fail("parallel vs sequential pipeline(%s, %s) image %dx%d diff=%d",
                              fa->name, fb->name, w, h, diff);
                    ok = 0;
                }
            }
            if (ok)
                test_pass();
        }
    }
}

static void test_strategies_agree(void)
{
    static const Strategy strategies[] = {STRATEGY_PIXEL, STRATEGY_ROW, STRATEGY_COL,
                                          STRATEGY_TILE};
    static const char *strategy_names[] = {"pixel", "row", "col", "tile"};
    static const int n_strategies = 4;
    static const int ch_counts[] = {1, 3, 4};
    static const int n_ch = 3;

    for (int ri = 0; ri < g_registry_count; ++ri)
    {
        const Filter *f = &g_registry[ri];

        for (int ci = 0; ci < n_ch; ++ci)
        {
            int ch = ch_counts[ci];

            for (int di = 0; g_test_dims[di][0]; ++di)
            {
                int w = g_test_dims[di][0], h = g_test_dims[di][1];

                stbi_uc *img =
                    img_random(w, h, ch, (unsigned)(ri * 300 + ci * 30 + w + h + time(NULL)));
                stbi_uc *ref = img_alloc(w, h, ch, 0);

                ConvArgs args = {img, ref, w, h, ch, f->kernel, f->kernel_size, f->factor, f->bias};

                convolve_sequential(&args);

                int all_ok = 1;
                for (int si = 0; si < n_strategies && all_ok; ++si)
                {
                    test_begin("strategy=%s filter=%s ch=%d %dx%d", strategy_names[si], f->name, ch,
                               w, h);
                    const Filter *steps[1] = {f};
                    stbi_uc *out = paralell_run(img, w, h, ch, steps, 1, strategies[si]);

                    int diff = img_max_diff(ref, out, w, h, ch);
                    free(out);

                    if (diff != 0)
                    {
                        test_fail("strategy %s vs ref: max diff=%d", strategy_names[si], diff);
                        all_ok = 0;
                    }
                    else
                    {
                        test_pass();
                    }
                }
                free(img);
                free(ref);
            }
        }
    }
}

int main(void)
{
    srand(time(NULL));

    test_zero_kernel();
    test_identity_kernel();
    test_zero_padding_invariance();
    test_compositionality();
    test_strategies_agree();

    printf("\n===============\n");
    printf("Results: %d/%d passed", g_passed, g_total);

    if (g_failed > 0)
        printf(", %d FAILED", g_failed);
    printf("\n");

    return g_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
