#include "semihost.h"

static int sh_call(int op, void *arg) {
    register int r0 __asm__("r0") = op;
    register void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

static int str_len(const char *s) { int n = 0; while (s[n]) n++; return n; }

int sh_open(const char *name) {
    int args[3] = { (int)name, 1 /* "rb" */, str_len(name) };
    return sh_call(0x01, args);
}

int sh_read(int fd, void *buf, int len) {
    int args[3] = { fd, (int)buf, len };
    return sh_call(0x06, args);
}

void sh_puts(const char *s) { sh_call(0x04, (void *)s); }

void sh_put_int(long v) {
    char b[24]; int i = 23; b[i] = 0;
    int neg = v < 0; unsigned long u = neg ? (unsigned long)(-v) : (unsigned long)v;
    do { b[--i] = (char)('0' + u % 10); u /= 10; } while (u);
    if (neg) b[--i] = '-';
    sh_puts(&b[i]);
}

void sh_exit(int code) {
    /* ADP_Stopped_ApplicationExit (0x20026); non-zero codes report an error */
    sh_call(0x18, (void *)(code == 0 ? 0x20026 : 0x20024));
    for (;;) {}
}
