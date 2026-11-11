/* tests/test_fastfp_fma.py's driver: the fast float regions' scalar multiply-add (a64_ff_fma32_fma32, FMA3) and multiply
   (a64_ff_mul32_fma32), each alone inside the regions' MXCSR window (a64_ff_begin / a64_ff_end), against the exact body
   the lifted code falls back to (recomp_fpe_fma32), value and FPSR. A run the window sends to its exact copy (a64_ff_end
   returning 0) is a fallback, always sound; a kept result must be Arm's bit for bit. Built at -march=x86-64-v3 only
   (without FMA3 the multiply-add is the exact helper itself).

   fastfp_fma_test fma <n> <seed>   n random triples, plus the edge list, for the multiply-add
   fastfp_fma_test mul <n> <seed>   the same for the multiply (FMUL: the addend is a zero of the product's sign)
   Exit 0: no mismatch, and both kept and fallen-back cases seen. 1: a mismatch. 3: the CPU has no FMA3/AVX2. */
#include "a64.h"
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <cpuid.h>
#include "a64_fastfp.h"
#if RECOMP_FPX_HOST != 2 || !defined(__FMA__)
#error "build at -march=x86-64-v3 (FMA3)"
#endif

static uint64_t rs;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static uint32_t r32(void) { return (uint32_t)(rnd() >> 16); }
static uint32_t rbelow(uint32_t n) { return (uint32_t)(((rnd() >> 32) * n) >> 32); }
static uint32_t mkf(uint32_t s, int e, uint32_t f) { return (s << 31) | ((uint32_t)e << 23) | (f & 0x7fffff); }
static float F(uint32_t x) { float f; memcpy(&f, &x, 4); return f; }
static uint32_t B(float f) { uint32_t x; memcpy(&x, &f, 4); return x; }

static uint32_t sparse(void) {   /* fractions with few bits: exact products, ties, cancellations */
    switch (rbelow(4)) { case 0: return 0; case 1: return (uint32_t)1 << rbelow(23); case 2: return r32() & 0x7ff000; default: return r32(); }
}
static uint32_t operand(void) {
    const uint32_t s = r32() & 1;
    switch (rbelow(14)) {
    case 0: return s << 31;                                    /* zero */
    case 1: return mkf(s, 0, r32() | 1);                       /* subnormal */
    case 2: return mkf(s, 1 + rbelow(3), sparse());            /* smallest normals */
    case 3: return mkf(s, 254 - rbelow(3), sparse());          /* largest normals */
    case 4: return mkf(s, 255, 0);                             /* inf */
    case 5: return mkf(s, 255, (r32() | 1) & 0x7fffff);        /* NaN, quiet or signalling */
    case 6: return mkf(s, 255, 0x400000 | r32());              /* quiet NaN */
    case 7: case 8: return mkf(s, 120 + rbelow(15), sparse()); /* near 1 */
    default: return mkf(s, 1 + rbelow(254), r32());           /* any normal */
    }
}
/* a product near the bottom of the normal range (rounding to and across 2^-126), or near the overflow edge */
static void edge_pair(uint32_t *a, uint32_t *b) {
    if (rbelow(4) == 0) {   /* a in [0.5, 1), b in [2^-126, 2^-124): products from 2^-127 up, many rounding to 2^-126 */
        *a = mkf(r32() & 1, 126, rbelow(2) ? (0x7fffff - rbelow(64)) : r32());
        *b = mkf(r32() & 1, 1 + rbelow(2), rbelow(2) ? (0x7fffff - rbelow(64)) : r32());
        return;
    }
    const int target = rbelow(8) == 0 ? 255 - (int)rbelow(3) : (int)rbelow(60) - 30;
    int ea = 1 + (int)rbelow(253), eb = target - ea + 127;
    if (eb < 1 || eb > 254) { eb = 1 + (int)rbelow(254); ea = target - eb + 127; }
    if (ea < 1) ea = 1;
    if (ea > 254) ea = 254;
    *a = mkf(r32() & 1, ea, rbelow(2) ? sparse() : r32());
    *b = mkf(r32() & 1, eb, rbelow(2) ? sparse() : r32());
}
static uint32_t addend(uint32_t a, uint32_t b) {
    const double p = (double)F(a) * (double)F(b);
    switch (rbelow(9)) {
    case 0: return operand();
    case 1: return B((float)-p);                                  /* near (often exact) cancellation */
    case 2: return B((float)-p) + (uint32_t)((int)rbelow(9) - 4); /* a few ulps off */
    case 3: return r32() & 0x80000000u;                           /* a zero */
    case 4: return mkf(r32() & 1, 0, r32());                      /* subnormal */
    case 5: return mkf(r32() & 1, 1 + rbelow(3), r32());          /* near the smallest normal */
    default: return mkf(r32() & 1, 1 + rbelow(254), r32());
    }
}

static uint64_t checks, kept, fell, mismatches;
static int report_left = 20;
static void check(uint64_t cr, uint64_t sr, uint32_t a, uint32_t b, uint32_t z, int mul) {
    Ctx ref, got;
    memset(&ref, 0, sizeof ref);
    ref.fpcr = cr; ref.fpsr = sr;
    got = ref;
    if (mul) z = (a ^ b) & 0x80000000u;   /* (FMUL as the lifter passes it) */
    const uint64_t r = recomp_fpe_fma32(&ref, a, b, z);
    const unsigned m = a64_ff_begin(&got);
    const uint64_t v = mul ? a64_ff_mul32_fma32(&got, a, b, z) : a64_ff_fma32_fma32(&got, a, b, z);
    const int ok = a64_ff_end(&got, m);
    checks++;
    if (!ok) { fell++; return; }
    kept++;
    if (v != r || got.fpsr != ref.fpsr) {
        mismatches++;
        if (report_left-- > 0)
            printf("MISMATCH %s a=%08x b=%08x z=%08x fpcr=%llx fpsr=%llx: exact %08llx/%llx fast %08llx/%llx\n",
                   mul ? "FMUL" : "FMLA", a, b, z, (unsigned long long)cr, (unsigned long long)sr, (unsigned long long)r,
                   (unsigned long long)ref.fpsr, (unsigned long long)v, (unsigned long long)got.fpsr);
    }
}
static void states(uint32_t a, uint32_t b, uint32_t z, int mul) {   /* FZ off/on, the flags IXC only or all set */
    for (int fz = 0; fz < 2; fz++) {
        check(fz ? 1u << 24 : 0, 0x10, a, b, z, mul);
        check(fz ? 1u << 24 : 0, 0x9f, a, b, z, mul);
    }
}

int main(int argc, char **argv) {
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx) || !(ecx >> 12 & 1) || !__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) ||
        !(ebx >> 5 & 1)) {
        printf("this CPU has no FMA/AVX2 (x86-64-v3)\n");
        return 3;
    }
    if (argc != 4 || (strcmp(argv[1], "fma") && strcmp(argv[1], "mul"))) {
        fprintf(stderr, "usage: fma|mul <n> <seed>\n");
        return 2;
    }
    const int mul = !strcmp(argv[1], "mul");
    /* the cases the window must send to the exact copy, and plain ones it must keep */
    static const uint32_t E[][3] = {
        {0x00000000u, 0x7f800000u, 0x7fc00001u},   /* 0 * inf + qNaN: Arm's default NaN and IOC */
        {0x7f800000u, 0x80000000u, 0xffc00002u},
        {0x3f000001u, 0x00fffffeu, 0x00000000u},   /* (1 - 2^-46) * 2^-126: rounds up to 2^-126, tiny before rounding */
        {0xbf000001u, 0x00fffffeu, 0x80000000u},
        {0x17800000u, 0x97800000u, 0x00800000u},   /* 2^-126 - 2^-160 */
        {0x3f800000u, 0x3f800000u, 0x00000123u},   /* a subnormal addend */
        {0x00000123u, 0x3f800000u, 0x3f800000u},   /* a subnormal operand */
        {0x3fc00000u, 0x40000000u, 0xc0400000u},   /* 1.5 * 2 - 3: an exact cancellation (+0) */
        {0x3f800001u, 0x3f800001u, 0xbf800000u},   /* inexact */
        {0x7f000000u, 0x7f000000u, 0x00000000u},   /* overflow */
        {0x7fc00003u, 0x3f800000u, 0x3f800000u},   /* quiet NaN operands */
        {0x3f800000u, 0x7fc00004u, 0x7fc00005u},
    };
    for (unsigned i = 0; i < sizeof E / sizeof E[0]; i++) states(E[i][0], E[i][1], E[i][2], mul);
    rs = strtoull(argv[3], 0, 0) * 0x9E3779B97F4A7C15ULL + 1;
    const uint64_t n = strtoull(argv[2], 0, 0);
    for (uint64_t i = 0; i < n; i++) {
        uint32_t a, b;
        if (rbelow(2)) edge_pair(&a, &b); else { a = operand(); b = operand(); }
        states(a, b, addend(a, b), mul);
    }
    printf("%s: %llu checks, %llu kept, %llu sent to the exact copy, %llu mismatches\n", argv[1],
           (unsigned long long)checks, (unsigned long long)kept, (unsigned long long)fell, (unsigned long long)mismatches);
    return mismatches ? 1 : (kept && fell) ? 0 : 4;
}
