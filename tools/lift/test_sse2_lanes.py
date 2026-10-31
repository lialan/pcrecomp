"""Execute lifted SIMD against integer/IEEE references, including aliasing,
saturation, wraparound, NaNs and over-width shifts. No arcade data needed."""
import math
import pathlib
import random
import struct
import subprocess
import tempfile
from test_bswap import emit


def signed(x, bits):
    return x - (1 << bits) if x & (1 << (bits - 1)) else x


def lanes(data, bits):
    return list(struct.unpack('<' + {16: '8H', 32: '4I', 64: '2Q'}[bits], data))


def packed(values, bits):
    return b''.join((v & ((1 << bits) - 1)).to_bytes(bits // 8, 'little') for v in values)


def reference(op, a, b):
    ah, bh = [list(map(lambda x: signed(x, 16), lanes(v, 16))) for v in (a, b)]
    if op in ('paddsw', 'psubsw', 'pmulhw', 'pmaxsw'):
        fn = {'paddsw': lambda x, y: x + y, 'psubsw': lambda x, y: x - y,
              'pmulhw': lambda x, y: (x * y) >> 16, 'pmaxsw': max}[op]
        return packed([max(-32768, min(32767, fn(x, y))) for x, y in zip(ah, bh)], 16)
    if op == 'pmaddwd':
        return packed([ah[i] * bh[i] + ah[i+1] * bh[i+1] for i in range(0, 8, 2)], 32)
    if op == 'pmuludq':
        x, y = lanes(a, 32), lanes(b, 32)
        return packed([x[i] * y[i] for i in (0, 2)], 64)
    if op == 'paddq':
        return packed([x + y for x, y in zip(lanes(a, 64), lanes(b, 64))], 64)
    if op == 'packssdw':
        values = [signed(x, 32) for v in (a, b) for x in lanes(v, 32)]
        return packed([max(-32768, min(32767, x)) for x in values], 16)
    if op in ('psrad', 'psrad_variable', 'psraw', 'pslld'):
        bits = 16 if op == 'psraw' else 32
        count = lanes(b, 64)[0] if op.endswith('variable') else {'psrad': 12, 'psraw': 16, 'pslld': 32}[op]
        if op == 'pslld': return packed([x << count if count < bits else 0 for x in lanes(a, bits)], bits)
        return packed([signed(x, bits) >> min(count, bits-1) for x in lanes(a, bits)], bits)
    if op == 'pshufhw':
        x = lanes(a, 16)
        return packed(x[:4] + [x[4+i] for i in (0, 2, 1, 3)], 16)
    if op == 'movshdup':
        x = lanes(a, 32)
        return packed([x[i] for i in (1, 1, 3, 3)], 32)
    if op.startswith('cmple'):
        bits, fmt = (64, '<2d') if op.endswith('pd') else (32, '<4f')
        return packed([-1 if x <= y else 0 for x, y in zip(struct.unpack(fmt, a), struct.unpack(fmt, b))], bits)
    raise AssertionError(op)


def main():
    encodings = dict(paddsw='660fedc1', psubsw='660fe9c1', pmulhw='660fe5c1',
                     pmaxsw='660feec1', pmaddwd='660ff5c1', pmuludq='660ff4c1',
                     paddq='660fd4c1', packssdw='660f6bc1', psrad='660f72e00c',
                     psrad_variable='660fe2c1', psraw='660f71e010', pslld='660f72f020',
                     pshufhw='f30f70c0d8', movshdup='f30f16c0',
                     cmpleps='0fc2c102', cmplepd='660fc2c102')
    rng = random.Random(917)
    vectors = [bytes(16), bytes([255])*16, packed([0x8000]*8, 16),
               packed([0x7fff, 0x8000]*4, 16),
               packed([0x7fc00000, 0xffc00000, 0, 0x80000000], 32),
               packed([0x7ff8000000000001, 0xfff0000000000000], 64)]
    vectors += [rng.getrandbits(128).to_bytes(16, 'little') for _ in range(40)]
    functions, tests = [], []
    for op, encoding in encodings.items():
        code = emit(bytes.fromhex(encoding))
        assert 'RECOMP_TODO' not in code, (op, code)
        functions.append('static void ' + op + '(CPU *c) { ' + code + ' }')
        pairs = [(a, b) for a in vectors[:6] for b in vectors[:6]]
        pairs += list(zip(vectors, reversed(vectors)))
        if op == 'psrad_variable':
            pairs += [(vectors[2], packed([n, 0], 64)) for n in (0, 1, 15, 16, 31, 32, 63, 64)]
        for a, b in pairs:
            expected = reference(op, a, b)
            blobs = [','.join(str(v) for v in blob) for blob in (a, b, expected)]
            tests.append('{ unsigned char a[16]={%s}, b[16]={%s}, expected[16]={%s};'
                         ' CPU c={0}; memcpy(&c.xmm[0],a,16); memcpy(&c.xmm[1],b,16);'
                         ' c.cf=c.zf=c.of=1; %s(&c);'
                         ' if(memcmp(&c.xmm[0],expected,16)||memcmp(&c.xmm[1],b,16)||!c.cf||!c.zf||!c.of)'
                         ' {fprintf(stderr,"%s case failed at %%d\\n",__LINE__);return 1;} }'
                         % (*blobs, op, op))
    for op, encoding in [('bsf', '0fbcc0'), ('bsr', '0fbdc0'), ('bsf16', '660fbcc0')]:
        code = emit(bytes.fromhex(encoding))
        assert 'RECOMP_TODO' not in code
        functions.append('static void ' + op + '(CPU *c) { ' + code + ' }')
        for value in [0, 0xffff0000] + [1 << i for i in range(32)]:
            bits = value & (0xffff if op == 'bsf16' else 0xffffffff)
            result = (bits.bit_length()-1 if op == 'bsr' else (bits & -bits).bit_length()-1) if bits else value
            if op == 'bsf16': result = (value & 0xffff0000) | (result & 0xffff)
            tests.append('{CPU c={0};c.eax=%su;c.cf=c.of=1;%s(&c);'
                         'if(c.eax!=%su||c.zf!=%d||!c.cf||!c.of)return 2;}'
                         % (value, op, result, not bool(bits)))
    source = '#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include "cpu.h"\n'
    source += '\n'.join(functions) + '\nint main(void){\n' + '\n'.join(tests) + '\nreturn 0;}\n'
    root = pathlib.Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as td:
        src, exe = pathlib.Path(td) / 'lanes.c', pathlib.Path(td) / 'lanes'
        src.write_text(source)
        subprocess.run(['cc', '-std=gnu11', '-O1', '-fsanitize=undefined', '-I',
                        str(root / 'runtime/recomp32_cpu'), str(src), '-lm', '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
    print('ok: %d executed SIMD and bit-scan cases with undefined-behavior checks' % len(tests))


if __name__ == '__main__':
    main()
