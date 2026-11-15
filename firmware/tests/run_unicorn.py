"""Run the Cortex-M4 firmware ELF in the Unicorn emulator (local check, no QEMU).
usage: python run_unicorn.py firmware.elf  (stream.bin in the current directory)"""
import sys, struct, time
from unicorn import Uc, UC_ARCH_ARM, UC_MODE_THUMB, UC_MODE_MCLASS, UC_HOOK_INTR, UcError
from unicorn import arm_const as A

d = open(sys.argv[1], 'rb').read()
phoff, = struct.unpack_from('<I', d, 0x1C); phentsize, phnum = struct.unpack_from('<HH', d, 0x2A)
uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
uc.ctl_set_cpu_model(A.UC_CPU_ARM_CORTEX_M4)
uc.mem_map(0x00000000, 0x100000); uc.mem_map(0x20000000, 0x40000); uc.mem_map(0xE0000000, 0x100000)
for i in range(phnum):
    p_type, off, vaddr, paddr, filesz, memsz, flags, align = struct.unpack_from('<8I', d, phoff + i * phentsize)
    if p_type == 1 and filesz:
        uc.mem_write(paddr, d[off:off + filesz])
sp, pc = struct.unpack('<II', bytes(uc.mem_read(0, 8)))
uc.reg_write(A.UC_ARM_REG_SP, sp)
try:
    uc.reg_write(A.UC_ARM_REG_CPACR, 0xF << 20)          # FPU on (startup also does this)
except Exception as e:
    print('note: cannot set CPACR directly:', e)
files, out = {}, []

def cstr(addr):
    b = bytearray()
    while True:
        c = uc.mem_read(addr, 1)[0]
        if c == 0: return b.decode()
        b.append(c); addr += 1

def on_intr(uc, intno, _):
    pc = uc.reg_read(A.UC_ARM_REG_PC)
    for a in (pc, pc - 2):
        if bytes(uc.mem_read(a, 2)) == b'\xab\xbe':          # bkpt 0xAB
            break
    else:
        print(f'unexpected exception {intno} at 0x{pc:08x}'); uc.emu_stop(); return
    op, arg = uc.reg_read(A.UC_ARM_REG_R0), uc.reg_read(A.UC_ARM_REG_R1)
    ret = 0
    if op == 0x04:
        s = cstr(arg); out.append(s); sys.stdout.write(s)
    elif op == 0x01:
        name_p, mode, ln = struct.unpack('<3I', bytes(uc.mem_read(arg, 12)))
        try: fd = len(files) + 3; files[fd] = open(cstr(name_p), 'rb'); ret = fd
        except OSError: ret = 0xFFFFFFFF
    elif op == 0x06:
        fd, buf, ln = struct.unpack('<3I', bytes(uc.mem_read(arg, 12)))
        data = files[fd].read(ln); uc.mem_write(buf, data); ret = ln - len(data)
    elif op == 0x18:
        print(f'[exit reason 0x{arg:x}]'); uc.emu_stop(); return
    uc.reg_write(A.UC_ARM_REG_R0, ret & 0xFFFFFFFF)
    uc.reg_write(A.UC_ARM_REG_PC, (a + 2) | 1)

uc.hook_add(UC_HOOK_INTR, on_intr)
t0 = time.time()
try:
    uc.emu_start(pc | 1, 0xFFFFFFFF)
except UcError as e:
    print('emulation error:', e, 'pc=0x%08x' % uc.reg_read(A.UC_ARM_REG_PC))
print(f'[emulated in {time.time() - t0:.1f} s]')
