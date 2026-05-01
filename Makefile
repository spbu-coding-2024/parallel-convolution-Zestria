CC = gcc
CFLAGS = -O2 -Wall -Wextra -Wpedantic -Wno-unused-function -std=c99
LDFLAGS = -lm -fopenmp

.PHONY: $(NON_PARALLEL) clean

sequential_conv: sequential_conv.c sequential_conv.h conv_common.h
	$(CC) $(CFLAGS) sequential_conv.c -o $@ $(LDFLAGS)

test_sequential_conv: test_sequential_conv.c sequential_conv.c
	$(CC) $(CFLAGS) test_sequential_conv.c -o $@ $(LDFLAGS)

clean:
	rm -f sequential_conv test_sequential_conv
