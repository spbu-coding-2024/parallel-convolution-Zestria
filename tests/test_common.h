#ifndef TEST_COMMON_H
#define TEST_COMMON_H

static int failures = 0;
static int checks = 0;

#define CHECK(cond)\
    {\
        ++checks;\
        if (!(cond)) {\
            fprintf(stderr, "FAIL: (%s:%d)\n", __FILE__, __LINE__);\
            ++failures;\
        }\
    }

#define TEST_REPORT(name) \
    do {\
        if (failures == 0) {\
            printf("%s: OK\n", name);\
            return 0;\
        }\
        fprintf(stderr, "%s: %d checks, %d failures\n", name, checks, failures);\
        return 1;\
    } while (0)

#endif /* TEST_COMMON_H */

