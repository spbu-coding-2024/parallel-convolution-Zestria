#include <stdio.h>
#include <string.h>

#include <convolution/convolution.h>
#include <convolution/kernels.h>

#include "../test_common.h"

static void test_identity_preserves_pixels(void) {
    uint8_t in_data[9] = {1,2,3,4,5,6,7,8,9};
    uint8_t out_data[9] = {0};
    conv_image in = {3, 3, in_data};
    conv_image out = { 3, 3, out_data};
    
    conv_kernel k = KERNEL_IDENTITY_3x3;
    CHECK(conv_apply_gray(&in, &out, &k) == CONV_OK); 
    CHECK(memcmp(in_data, out_data, 9) == 0)
}

int main(void) {
    test_identity_preserves_pixels();
    
    TEST_REPORT("test_sequential_convolution"); 
    return failures == 0 ? 0 : 1;
}

