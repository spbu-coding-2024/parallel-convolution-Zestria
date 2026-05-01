# Parallel Image Convolution

An extension of `sequential_conv` that exposes four OpenMP parallelisation **strategies** for image convolution, making it easy to benchmark and compare different work-decomposition approaches on your hardware.

## Strategies

| Flag | Parallelised over |
|---|---|
| `pixel` | Individual pixels (flat index) |
| `row` | Image rows |
| `col` | Image columns |
| `tile` | Rectangular tiles |

## Usage

```sh
./parallel_conv <input.png> <output.png> <strategy> [tile_w tile_h] --filter <filter1> [filter2 ...]
```

Examples:

```sh
# Row-parallel gaussian blur
./parallel_conv photo.png out.png row --filter gaussian

# Tile-parallel sharpen, 128×128 tiles
./parallel_conv photo.png out.png tile 128 128 --filter sharpen

# Multi-step pipeline: sharpen then blur, column strategy
./parallel_conv photo.png out.png col --filter sharpen gaussian
```

## Build

```sh
make parallel_conv
```

Requires `stb_image` and `stb_image_write` (single-file headers, place alongside the source).

Control thread count via the environment:

```sh
OMP_NUM_THREADS=8 ./parallel_conv ...
```

## Testing

```sh
make test_parallel_conv && ./test_parallel_conv
```