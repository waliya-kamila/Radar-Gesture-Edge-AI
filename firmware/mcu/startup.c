/* Cortex-M4 start-up: vector table, .data/.bss init, FPU enable. */
#include <stdint.h>
#include "semihost.h"

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
int main(void);

void Reset_Handler(void) {
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;
    for (dst = &_sbss; dst < &_ebss;) *dst++ = 0;
    *(volatile uint32_t *)0xE000ED88 |= (0xFu << 20);     /* CPACR: enable CP10/CP11 (FPU) */
    __asm__ volatile("dsb\n isb");
    sh_exit(main());
}

void Default_Handler(void) {
    sh_puts("FAULT: unexpected exception\r\n");
    sh_exit(2);
}

__attribute__((section(".isr_vector"), used))
const void *const vector_table[16] = {
    &_estack, Reset_Handler,
    Default_Handler, Default_Handler, Default_Handler, Default_Handler, Default_Handler,
    0, 0, 0, 0,
    Default_Handler, Default_Handler, 0, Default_Handler, Default_Handler,
};
