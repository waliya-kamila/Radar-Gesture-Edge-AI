/* Memory budget of a mid-range Cortex-M4 MCU: 512 KB flash, 128 KB SRAM.
 * Addresses follow the Arm MPS2 AN386 board emulated by QEMU. The link fails
 * if code + data do not fit, which is itself a check of the memory budget. */
MEMORY
{
  FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 512K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}
_estack = ORIGIN(RAM) + LENGTH(RAM);
ENTRY(Reset_Handler)

SECTIONS
{
  .isr_vector : { KEEP(*(.isr_vector)) } > FLASH
  .text : { *(.text*) *(.rodata*) . = ALIGN(4); } > FLASH
  .ARM.exidx : { *(.ARM.exidx*) } > FLASH
  _sidata = LOADADDR(.data);
  .data : { _sdata = .; *(.data*) . = ALIGN(4); _edata = .; } > RAM AT> FLASH
  .bss (NOLOAD) : { _sbss = .; *(.bss*) *(COMMON) . = ALIGN(4); _ebss = .; } > RAM
  .stack (NOLOAD) : { . = ALIGN(8); . = . + 8K; } > RAM
}
