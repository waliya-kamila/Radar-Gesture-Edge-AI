/* Minimal ARM semihosting: lets the simulated MCU print text and read the
 * recorded radar stream from a file on the host (stands in for UART + sensor). */
#ifndef SEMIHOST_H
#define SEMIHOST_H
int  sh_open(const char *name);                 /* returns handle, or -1      */
int  sh_read(int fd, void *buf, int len);       /* returns bytes NOT read     */
void sh_puts(const char *s);
void sh_put_int(long v);
void sh_exit(int code);
#endif
