#!/bin/sh
# Build the Cortex-M4F firmware. Needs only gcc-arm-none-eabi (no libc/libm).
set -e
cd "$(dirname "$0")"
CFLAGS="-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -std=c99 -Wall \
        -ffreestanding -fno-builtin -fno-tree-loop-distribute-patterns -fno-math-errno \
        -ffunction-sections -fdata-sections -Isrc -Imcu"
arm-none-eabi-gcc $CFLAGS src/radar_pipeline.c mcu/semihost.c mcu/startup.c mcu/main.c mcu/libc_min.c \
    -T mcu/mps2_an386.ld -nostdlib -Wl,--gc-sections -Wl,-Map=firmware.map -lgcc -o firmware.elf
echo "--- memory used (bytes) ---"
arm-none-eabi-size firmware.elf
