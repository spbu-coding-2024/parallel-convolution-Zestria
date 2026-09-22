#ifndef CONVOLUTION_IO_H
#define CONVOLUTION_IO_H

#include <convolution/convolution.h>
#include <convolution/image.h>

/* Read PNG and load it in grayscale. */
conv_status conv_io_load_png(char *path, conv_image **out);

/* Save grayscale PNG to file */
conv_status conv_io_save_png(char *path, conv_image *out);

#endif /* CONVOLUTION_IO_H */

