# Image Convolution

A C image-processing tool that applies a chain of convolution filters to PNG images. Processes pixels using a configurable filter pipeline.

## Usage

```sh
./sequential_conv <input.png> <output.png> --filter <filter1> [filter2 ...]
```

Example — sharpen then blur:
```sh
./sequential_conv photo.png result.png --filter sharpen gaussian
```

To list available filters, run with an unknown filter name; the registry will print all options.

## Build

```sh
make sequential_conv
```

Requires `stb_image` and `stb_image_write` headers (single-file libraries, drop them in the same directory).

## Testing

```sh
make test_sequential_conv && ./test_sequential_conv
```