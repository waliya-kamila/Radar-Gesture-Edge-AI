/* The two C library routines the compiler may call implicitly; the firmware
 * needs no other libc or libm function. */
#include <stddef.h>

void *memcpy(void *dst, const void *src, size_t n) {
    unsigned char *d = dst; const unsigned char *s = src;
    while (n--) *d++ = *s++;
    return dst;
}

void *memset(void *dst, int v, size_t n) {
    unsigned char *d = dst;
    while (n--) *d++ = (unsigned char)v;
    return dst;
}
