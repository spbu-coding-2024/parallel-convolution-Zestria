#ifndef CONVOLUTION_KERNELS_H
#define CONVOLUTION_KERNELS_H

#include <convolution/convolution.h>

/*
 * Allocate a kernel whose coefficients are zero filled, with factor 1.0 and bias 0.0.
 * Returns NULL for non-positive dimensions or when an allocation fails.
 */
conv_kernel *conv_kernel_create(int width, int height);

/*
 * Free a kernel returned by conv_kernel_create()
 * Nothing happens for NULL
 */
void conv_kernel_destroy(conv_kernel *kernel);

/*
 * Preset kernels live in static, non-owned storage. Every preset satisfies
 * conv_kernel_validate() and must never be passed to conv_kernel_destroy().
 */
extern const conv_kernel KERNEL_IDENTITY_3x3;
extern const conv_kernel KERNEL_BLUR_3x3;
extern const conv_kernel KERNEL_BLUR_5x5;
extern const conv_kernel KERNEL_GAUSSIAN_BLUR_3x3;
extern const conv_kernel KERNEL_MOTION_BLUR_9x9;
extern const conv_kernel KERNEL_FIND_HORIZONTAL_EDGES_5x5;
extern const conv_kernel KERNEL_FIND_VERTICAL_EDGES_5x5;
extern const conv_kernel KERNEL_FIND_DIAGONAL_EDGES_5x5;
extern const conv_kernel KERNEL_FIND_ALL_EDGES_3x3;
extern const conv_kernel KERNEL_SHARPEN_3x3;
extern const conv_kernel KERNEL_SHARPEN_5x5;
extern const conv_kernel KERNEL_SHARPEN_EXCESSIVELY_3x3;
extern const conv_kernel KERNEL_EMBOSS_3x3;
extern const conv_kernel KERNEL_EMBOSS_5x5;
extern const conv_kernel KERNEL_MEAN_3x3;

#endif /* CONVOLUTION_KERNELS_H */

