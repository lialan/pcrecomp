"""Execute lifted MOVMSKPS for every sign mask, including NaNs and -0."""
import pathlib
import subprocess
import tempfile
from test_bswap import emit


def main():
    code = emit(bytes.fromhex("0f50d9"))  # movmskps ebx,xmm1
    assert "RECOMP_TODO" not in code, code
    source = r'''
#include <string.h>
#include <stdlib.h>
#include "cpu.h"
static void lifted(CPU *c) { CODE }
int main(void) {
    for (unsigned mask = 0; mask < 16; mask++) {
        CPU c = {0};
        c.ebx = 0xffffffffu;
        c.cf = c.zf = c.of = 1;
        for (unsigned lane = 0; lane < 4; lane++)
            c.xmm[1].u32[lane] = ((mask >> lane) & 1u) * 0x80000000u
                                | (lane < 2 ? 0 : 0x7fc01234u);
        XMM original = c.xmm[1];
        lifted(&c);
        if (c.ebx != mask || !c.cf || !c.zf || !c.of ||
            memcmp(&original, &c.xmm[1], sizeof original)) return 1;
    }
    return 0;
}
'''.replace("CODE", code)
    root = pathlib.Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        src = pathlib.Path(td) / "test.c"
        exe = pathlib.Path(td) / "test"
        src.write_text(source)
        subprocess.run(["cc", "-std=gnu11", "-O2", "-I", str(root / "runtime/recomp32_cpu"),
                        str(src), "-lm", "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
    print("movmskps: all sign masks, NaNs, negative zero, source and flags preserved")


if __name__ == "__main__":
    main()
