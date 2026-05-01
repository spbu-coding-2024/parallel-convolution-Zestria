CC = gcc
CFLAGS = -O2 -Wall -Wextra -Wpedantic -Wno-unused-function -std=c99
LDFLAGS = -lm -fopenmp

.PHONY: $(NON_PARALLEL) clean

sequential_conv: sequential_conv.c sequential_conv.h conv_common.h
	$(CC) $(CFLAGS) sequential_conv.c -o $@ $(LDFLAGS)

test_sequential_conv: test_sequential_conv.c sequential_conv.c
	$(CC) $(CFLAGS) test_sequential_conv.c -o $@ $(LDFLAGS)

parallel_conv: parallel_conv.c conv_common.h
	$(CC) $(CFLAGS) parallel_conv.c -o $@ $(LDFLAGS)

test_parallel_conv: test_parallel_conv.c parallel_conv.c sequential_conv.c sequential_conv.h conv_common.h
	$(CC) $(CFLAGS) -DCONV_TEST_BUILD sequential_conv.c test_parallel_conv.c -o $@ $(LDFLAGS)

clean:
	rm -f test_parallel_conv sequential_conv parallel_conv test_sequential_conv
