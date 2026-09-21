#include <stdio.h>
#include <string.h>

#include <convolution/convolution.h>
#include <convolution/kernels.h>

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg)\
    {\
        ++checks;\
        if (!(cond)) {\
            fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);\
            ++failures;\
        }\
    }

#define CHECK_STATUS(expr, expected, msg)\
    {\
        conv_status _a = (expr);\
        conv_status _e = (expected);\
        ++checks;\
        if (_a != _e) { \
            fprintf(stderr, "FAIL: %s - expected %d, got %d (%s:%d)\n", msg, (int)_e, (int)_a, __FILE__, __LINE__);\
            ++failures;\
        }\
    }

static void test_identity_preserves_pixels(void) {
    uint8_t in_data[9] = {1,2,3,4,5,6,7,8,9};
    uint8_t out_data[9] = {0};
    conv_image in = {3, 3, in_data};
    conv_image out = { 3, 3, out_data};
    
    conv_kernel k = KERNEL_IDENTITY_3x3;
    CHECK_STATUS(conv_apply_gray(&in, &out, &k), CONV_OK, "identity: status"); 
    CHECK(memcmp(in_data, out_data, 9) == 0, "identity: pixels preserved")
}

int main(void) {
    test_identity_preserves_pixels();
    
    printf("\n%d checks, %d failures\n", checks, failures);    
    return failures == 0 ? 0 : 1;
}

