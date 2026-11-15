/* Host (PC) implementation of the semihosting calls, so the exact firmware
 * main.c can also be run natively: same output, no emulator needed. */
#include <stdio.h>
#include <stdlib.h>
#include "semihost.h"

static FILE *files[8];

int sh_open(const char *name) {
    for (int i = 1; i < 8; i++)
        if (!files[i]) { files[i] = fopen(name, "rb"); return files[i] ? i : -1; }
    return -1;
}
int sh_read(int fd, void *buf, int len) { return len - (int)fread(buf, 1, (size_t)len, files[fd]); }
void sh_puts(const char *s) { fputs(s, stdout); }
void sh_put_int(long v) { printf("%ld", v); }
void sh_exit(int code) { fflush(stdout); exit(code); }

int main(void);
