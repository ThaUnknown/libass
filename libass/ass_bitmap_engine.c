/*
 * Copyright (C) 2021-2022 libass contributors
 *
 * This file is part of libass.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include "config.h"
#include "ass_compat.h"

#include <stdbool.h>

#include "ass_bitmap_engine.h"
#include "x86/cpuid.h"

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#include <string.h>
#include "ass_rasterizer.h"   // struct segment, SEGFLAG_* for the rasterizer port
#ifndef ALIGNMENT
#define ALIGNMENT 16
#endif
#ifndef ASSUME
#define ASSUME(x) ((void)0)
#endif

/*
 * SIMD128 ports of the libass kernels that otherwise run scalar on wasm32
 * (libass ships no wasm SIMD). Every kernel below is a bit-exact port of its
 * `_c` reference, validated byte-for-byte by the fuzz harnesses under build/.
 *
 * Blend kernels: 16 bytes/iter with a scalar tail for width % 16.
 */

void ass_add_bitmaps_wasm(uint8_t *restrict dst, ptrdiff_t dst_stride,
                          const uint8_t *restrict src, ptrdiff_t src_stride,
                          size_t width, size_t height)
{
    size_t w16 = width & ~(size_t) 15;
    uint8_t *end = dst + dst_stride * height;
    while (dst < end) {
        size_t x = 0;
        for (; x < w16; x += 16) {
            // saturating 8-bit add == FFMIN(dst+src, 255)
            v128_t d = wasm_v128_load(dst + x);
            v128_t s = wasm_v128_load(src + x);
            wasm_v128_store(dst + x, wasm_u8x16_add_sat(d, s));
        }
        for (; x < width; x++) {
            unsigned out = dst[x] + src[x];
            dst[x] = out < 255 ? out : 255;
        }
        dst += dst_stride;
        src += src_stride;
    }
}

// Shared helper: (a * b + 255) >> 8 over 16 bytes, narrowed back to u8.
static inline v128_t mul255_round(v128_t a, v128_t b)
{
    const v128_t c255 = wasm_i16x8_splat(255);
    v128_t lo = wasm_u16x8_extmul_low_u8x16(a, b);
    v128_t hi = wasm_u16x8_extmul_high_u8x16(a, b);
    lo = wasm_u16x8_shr(wasm_i16x8_add(lo, c255), 8);
    hi = wasm_u16x8_shr(wasm_i16x8_add(hi, c255), 8);
    return wasm_u8x16_narrow_i16x8(lo, hi);
}

void ass_imul_bitmaps_wasm(uint8_t *restrict dst, ptrdiff_t dst_stride,
                           const uint8_t *restrict src, ptrdiff_t src_stride,
                           size_t width, size_t height)
{
    const v128_t c255 = wasm_u8x16_splat(255);
    size_t w16 = width & ~(size_t) 15;
    uint8_t *end = dst + dst_stride * height;
    while (dst < end) {
        size_t x = 0;
        for (; x < w16; x += 16) {
            // dst = (dst * (255 - src) + 255) >> 8
            v128_t d = wasm_v128_load(dst + x);
            v128_t inv = wasm_u8x16_sub_sat(c255, wasm_v128_load(src + x));
            wasm_v128_store(dst + x, mul255_round(d, inv));
        }
        for (; x < width; x++)
            dst[x] = (dst[x] * (255 - src[x]) + 255) >> 8;
        dst += dst_stride;
        src += src_stride;
    }
}

void ass_mul_bitmaps_wasm(uint8_t *restrict dst, ptrdiff_t dst_stride,
                          const uint8_t *restrict src1, ptrdiff_t src1_stride,
                          const uint8_t *restrict src2, ptrdiff_t src2_stride,
                          size_t width, size_t height)
{
    size_t w16 = width & ~(size_t) 15;
    uint8_t *end = dst + dst_stride * height;
    while (dst < end) {
        size_t x = 0;
        for (; x < w16; x += 16) {
            // dst = (src1 * src2 + 255) >> 8
            v128_t a = wasm_v128_load(src1 + x);
            v128_t b = wasm_v128_load(src2 + x);
            wasm_v128_store(dst + x, mul255_round(a, b));
        }
        for (; x < width; x++)
            dst[x] = (src1[x] * src2[x] + 255) >> 8;
        dst  += dst_stride;
        src1 += src1_stride;
        src2 += src2_stride;
    }
}

/*
 * Gaussian blur, alignment-16 / STRIPE_WIDTH-8 variant (selected when
 * ASS_FLAG_WIDE_STRIPE is unset). Vertical class: unpack, pack, and the
 * vertical shrink/expand/blur passes operate on whole stripes at a fixed lane
 * offset, so they map to one i16x8 (or a pair of i32x4 where sums exceed
 * int16). The horizontal class follows further down.
 */
#define SW 8  // STRIPE_WIDTH for alignment 16

static int16_t zero_line_w[SW];

static inline const int16_t *get_line_w(const int16_t *ptr, size_t offs, size_t size)
{
    return offs < size ? ptr + offs : zero_line_w;
}

void ass_stripe_unpack16_wasm(int16_t *restrict dst, const uint8_t *restrict src,
                              ptrdiff_t src_stride, size_t width, size_t height)
{
    const v128_t one = wasm_i16x8_splat(1);
    for (size_t y = 0; y < height; y++) {
        int16_t *ptr = dst;
        for (size_t x = 0; x < width; x += SW) {
            // ptr[k] = (uint16_t)(((src<<7)|(src>>1)) + 1) >> 1
            v128_t s = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(src + x));
            v128_t v = wasm_v128_or(wasm_i16x8_shl(s, 7), wasm_u16x8_shr(s, 1));
            v = wasm_u16x8_shr(wasm_i16x8_add(v, one), 1);
            wasm_v128_store(ptr, v);
            ptr += SW * height;
        }
        dst += SW;
        src += src_stride;
    }
}

void ass_stripe_pack16_wasm(uint8_t *restrict dst, ptrdiff_t dst_stride,
                            const int16_t *restrict src, size_t width, size_t height)
{
    const v128_t dither_even = wasm_i16x8_make(8, 40, 8, 40, 8, 40, 8, 40);
    const v128_t dither_odd  = wasm_i16x8_make(56, 24, 56, 24, 56, 24, 56, 24);
    for (size_t x = 0; x < width; x += SW) {
        uint8_t *ptr = dst;
        for (size_t y = 0; y < height; y++) {
            // ptr[k] = (uint16_t)(src - (src>>8) + dither) >> 6
            v128_t s = wasm_v128_load(src);
            v128_t v = wasm_i16x8_sub(s, wasm_i16x8_shr(s, 8));
            v = wasm_i16x8_add(v, (y & 1) ? dither_odd : dither_even);
            v = wasm_u16x8_shr(v, 6);
            wasm_v128_store64_lane(ptr, wasm_u8x16_narrow_i16x8(v, v), 0);
            ptr += dst_stride;
            src += SW;
        }
        dst += SW;
    }
    size_t left = dst_stride - ((width + SW - 1) & ~(size_t)(SW - 1));
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < left; x++)
            dst[x] = 0;
        dst += dst_stride;
    }
}

// shrink_func over an i32x4 lane group (intermediates exceed int16 range)
static inline v128_t shrink4(v128_t p1p, v128_t p1n, v128_t z0p,
                             v128_t z0n, v128_t n1p, v128_t n1n)
{
    v128_t r = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_add(p1p, p1n),
                                             wasm_i32x4_add(n1p, n1n)), 1);
    r = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_add(r, z0p), z0n), 1);
    r = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_add(r, p1n), n1p), 1);
    r = wasm_i32x4_add(wasm_i32x4_add(wasm_i32x4_add(r, z0p), z0n),
                       wasm_i32x4_splat(2));
    return wasm_i32x4_shr(r, 2);
}

void ass_shrink_vert16_wasm(int16_t *restrict dst, const int16_t *restrict src,
                            size_t src_width, size_t src_height)
{
    size_t dst_height = (src_height + 5) >> 1;
    size_t step = SW * src_height;
    for (size_t x = 0; x < src_width; x += SW) {
        size_t offs = 0;
        for (size_t y = 0; y < dst_height; y++) {
            v128_t p1p = wasm_v128_load(get_line_w(src, offs - 4 * SW, step));
            v128_t p1n = wasm_v128_load(get_line_w(src, offs - 3 * SW, step));
            v128_t z0p = wasm_v128_load(get_line_w(src, offs - 2 * SW, step));
            v128_t z0n = wasm_v128_load(get_line_w(src, offs - 1 * SW, step));
            v128_t n1p = wasm_v128_load(get_line_w(src, offs - 0 * SW, step));
            v128_t n1n = wasm_v128_load(get_line_w(src, offs + 1 * SW, step));
            v128_t lo = shrink4(wasm_i32x4_extend_low_i16x8(p1p), wasm_i32x4_extend_low_i16x8(p1n),
                                wasm_i32x4_extend_low_i16x8(z0p), wasm_i32x4_extend_low_i16x8(z0n),
                                wasm_i32x4_extend_low_i16x8(n1p), wasm_i32x4_extend_low_i16x8(n1n));
            v128_t hi = shrink4(wasm_i32x4_extend_high_i16x8(p1p), wasm_i32x4_extend_high_i16x8(p1n),
                                wasm_i32x4_extend_high_i16x8(z0p), wasm_i32x4_extend_high_i16x8(z0n),
                                wasm_i32x4_extend_high_i16x8(n1p), wasm_i32x4_extend_high_i16x8(n1n));
            wasm_v128_store(dst, wasm_i16x8_narrow_i32x4(lo, hi));
            dst  += SW;
            offs += 2 * SW;
        }
        src += step;
    }
}

void ass_expand_vert16_wasm(int16_t *restrict dst, const int16_t *restrict src,
                            size_t src_width, size_t src_height)
{
    const v128_t one = wasm_i16x8_splat(1);
    size_t dst_height = 2 * src_height + 4;
    size_t step = SW * src_height;
    for (size_t x = 0; x < src_width; x += SW) {
        size_t offs = 0;
        for (size_t y = 0; y < dst_height; y += 2) {
            v128_t p1 = wasm_v128_load(get_line_w(src, offs - 2 * SW, step));
            v128_t z0 = wasm_v128_load(get_line_w(src, offs - 1 * SW, step));
            v128_t n1 = wasm_v128_load(get_line_w(src, offs - 0 * SW, step));
            // expand_func: all u16 arithmetic, values stay < 0x8000
            v128_t r = wasm_u16x8_shr(wasm_i16x8_add(wasm_u16x8_shr(wasm_i16x8_add(p1, n1), 1), z0), 1);
            v128_t rp = wasm_u16x8_shr(wasm_i16x8_add(wasm_i16x8_add(wasm_u16x8_shr(wasm_i16x8_add(r, p1), 1), z0), one), 1);
            v128_t rn = wasm_u16x8_shr(wasm_i16x8_add(wasm_i16x8_add(wasm_u16x8_shr(wasm_i16x8_add(r, n1), 1), z0), one), 1);
            wasm_v128_store(dst, rp);
            wasm_v128_store(dst + SW, rn);
            dst  += 2 * SW;
            offs += 1 * SW;
        }
        src += step;
    }
}

static inline void blur_vert_wasm(int16_t *restrict dst, const int16_t *restrict src,
                                  size_t src_width, size_t src_height,
                                  const int16_t *restrict param, const int n)
{
    size_t dst_height = src_height + 2 * n;
    size_t step = SW * src_height;
    for (size_t x = 0; x < src_width; x += SW) {
        size_t offs = 0;
        for (size_t y = 0; y < dst_height; y++) {
            v128_t acc_lo = wasm_i32x4_splat(0x8000);
            v128_t acc_hi = acc_lo;
            v128_t center = wasm_v128_load(get_line_w(src, offs - n * SW, step));
            for (int i = n; i > 0; i--) {
                v128_t l1 = wasm_v128_load(get_line_w(src, offs - (n + i) * SW, step));
                v128_t l2 = wasm_v128_load(get_line_w(src, offs - (n - i) * SW, step));
                v128_t pv = wasm_i16x8_splat(param[i - 1]);
                v128_t d1 = wasm_i16x8_sub(l1, center);
                v128_t d2 = wasm_i16x8_sub(l2, center);
                acc_lo = wasm_i32x4_add(acc_lo, wasm_i32x4_add(
                             wasm_i32x4_extmul_low_i16x8(d1, pv),
                             wasm_i32x4_extmul_low_i16x8(d2, pv)));
                acc_hi = wasm_i32x4_add(acc_hi, wasm_i32x4_add(
                             wasm_i32x4_extmul_high_i16x8(d1, pv),
                             wasm_i32x4_extmul_high_i16x8(d2, pv)));
            }
            // dst[k] = center[k] + (acc[k] >> 16)
            v128_t lo = wasm_i32x4_add(wasm_i32x4_extend_low_i16x8(center), wasm_i32x4_shr(acc_lo, 16));
            v128_t hi = wasm_i32x4_add(wasm_i32x4_extend_high_i16x8(center), wasm_i32x4_shr(acc_hi, 16));
            wasm_v128_store(dst, wasm_i16x8_narrow_i32x4(lo, hi));
            dst  += SW;
            offs += SW;
        }
        src += step;
    }
}

void ass_blur4_vert16_wasm(int16_t *restrict dst, const int16_t *restrict src,
                           size_t w, size_t h, const int16_t *restrict param)
{ blur_vert_wasm(dst, src, w, h, param, 4); }
void ass_blur5_vert16_wasm(int16_t *restrict dst, const int16_t *restrict src,
                           size_t w, size_t h, const int16_t *restrict param)
{ blur_vert_wasm(dst, src, w, h, param, 5); }
void ass_blur6_vert16_wasm(int16_t *restrict dst, const int16_t *restrict src,
                           size_t w, size_t h, const int16_t *restrict param)
{ blur_vert_wasm(dst, src, w, h, param, 6); }
void ass_blur7_vert16_wasm(int16_t *restrict dst, const int16_t *restrict src,
                           size_t w, size_t h, const int16_t *restrict param)
{ blur_vert_wasm(dst, src, w, h, param, 7); }
void ass_blur8_vert16_wasm(int16_t *restrict dst, const int16_t *restrict src,
                           size_t w, size_t h, const int16_t *restrict param)
{ blur_vert_wasm(dst, src, w, h, param, 8); }
#undef SW

// Gaussian blur, horizontal class (stripe-8 / align-16): shifted-load taps with
// even/odd deinterleave (shrink) and interleaved output (expand).
#define HSW 8

static int16_t hzero_line[HSW];

static inline const int16_t *hget_line(const int16_t *ptr, size_t offs, size_t size)
{
    return offs < size ? ptr + offs : hzero_line;
}
static inline void hcopy_line(int16_t *buf, const int16_t *ptr, size_t offs, size_t size)
{
    memcpy(buf, hget_line(ptr, offs, size), HSW * sizeof(buf[0]));
}

// shrink_func on an i32x4 lane group (intermediates exceed int16 range)
static inline v128_t hshrink4(v128_t p1p, v128_t p1n, v128_t z0p,
                              v128_t z0n, v128_t n1p, v128_t n1n)
{
    v128_t r = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_add(p1p, p1n),
                                             wasm_i32x4_add(n1p, n1n)), 1);
    r = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_add(r, z0p), z0n), 1);
    r = wasm_i32x4_shr(wasm_i32x4_add(wasm_i32x4_add(r, p1n), n1p), 1);
    r = wasm_i32x4_add(wasm_i32x4_add(wasm_i32x4_add(r, z0p), z0n),
                       wasm_i32x4_splat(2));
    return wasm_i32x4_shr(r, 2);
}

void ass_shrink_horz16_wasm(int16_t *restrict dst, const int16_t *restrict src,
                            size_t src_width, size_t src_height)
{
    size_t dst_width = (src_width + 5) >> 1;
    size_t size = ((src_width + 7) & ~(size_t)7) * src_height;
    size_t step = HSW * src_height;

    size_t offs = 0;
    _Alignas(16) int16_t buf[3 * HSW];
    int16_t *ptr = buf + HSW;
    for (size_t x = 0; x < dst_width; x += HSW) {
        for (size_t y = 0; y < src_height; y++) {
            hcopy_line(ptr - 1 * HSW, src, offs - 1 * step, size);
            hcopy_line(ptr + 0 * HSW, src, offs + 0 * step, size);
            hcopy_line(ptr + 1 * HSW, src, offs + 1 * step, size);

            // shrink_func taps ptr[2k-4..2k+1]: load ptr[-4..19], deinterleave
            // into even/odd lanes, then shift to gather the six taps per stripe.
            v128_t va = wasm_v128_load(ptr - 4);
            v128_t vb = wasm_v128_load(ptr + 4);
            v128_t vc = wasm_v128_load(ptr + 8);
            v128_t even_lo = wasm_i16x8_shuffle(va, vb, 0, 2, 4, 6, 8, 10, 12, 14);
            v128_t odd_lo  = wasm_i16x8_shuffle(va, vb, 1, 3, 5, 7, 9, 11, 13, 15);
            v128_t p1p = even_lo;
            v128_t z0p = wasm_i16x8_shuffle(even_lo, vc, 1, 2, 3, 4, 5, 6, 7, 12);
            v128_t n1p = wasm_i16x8_shuffle(even_lo, vc, 2, 3, 4, 5, 6, 7, 12, 14);
            v128_t p1n = odd_lo;
            v128_t z0n = wasm_i16x8_shuffle(odd_lo, vc, 1, 2, 3, 4, 5, 6, 7, 13);
            v128_t n1n = wasm_i16x8_shuffle(odd_lo, vc, 2, 3, 4, 5, 6, 7, 13, 15);

            v128_t lo = hshrink4(wasm_i32x4_extend_low_i16x8(p1p), wasm_i32x4_extend_low_i16x8(p1n),
                                 wasm_i32x4_extend_low_i16x8(z0p), wasm_i32x4_extend_low_i16x8(z0n),
                                 wasm_i32x4_extend_low_i16x8(n1p), wasm_i32x4_extend_low_i16x8(n1n));
            v128_t hi = hshrink4(wasm_i32x4_extend_high_i16x8(p1p), wasm_i32x4_extend_high_i16x8(p1n),
                                 wasm_i32x4_extend_high_i16x8(z0p), wasm_i32x4_extend_high_i16x8(z0n),
                                 wasm_i32x4_extend_high_i16x8(n1p), wasm_i32x4_extend_high_i16x8(n1n));
            wasm_v128_store(dst, wasm_i16x8_narrow_i32x4(lo, hi));

            dst  += HSW;
            offs += HSW;
        }
        offs += step;
    }
}

static inline void hexpand_block(int16_t *out0, int16_t *out1, const int16_t *ptr)
{
    const v128_t one = wasm_i16x8_splat(1);
    // p1[k]=ptr[k-2], z0[k]=ptr[k-1], n1[k]=ptr[k]
    v128_t va = wasm_v128_load(ptr - 2);
    v128_t vb = wasm_v128_load(ptr);
    v128_t p1 = va;
    v128_t z0 = wasm_i16x8_shuffle(va, vb, 1, 2, 3, 4, 5, 6, 7, 14);
    v128_t n1 = vb;

    // expand_func, all u16 arithmetic (values stay < 0x8000)
    v128_t r  = wasm_u16x8_shr(wasm_i16x8_add(wasm_u16x8_shr(wasm_i16x8_add(p1, n1), 1), z0), 1);
    v128_t rp = wasm_u16x8_shr(wasm_i16x8_add(wasm_i16x8_add(wasm_u16x8_shr(wasm_i16x8_add(r, p1), 1), z0), one), 1);
    v128_t rn = wasm_u16x8_shr(wasm_i16x8_add(wasm_i16x8_add(wasm_u16x8_shr(wasm_i16x8_add(r, n1), 1), z0), one), 1);

    // interleave even/odd outputs: out[2k]=rp[k], out[2k+1]=rn[k]
    v128_t lo = wasm_i16x8_shuffle(rp, rn, 0, 8, 1, 9, 2, 10, 3, 11);
    v128_t hi = wasm_i16x8_shuffle(rp, rn, 4, 12, 5, 13, 6, 14, 7, 15);
    wasm_v128_store(out0, lo);
    wasm_v128_store(out1, hi);
}

void ass_expand_horz16_wasm(int16_t *restrict dst, const int16_t *restrict src,
                            size_t src_width, size_t src_height)
{
    size_t dst_width = 2 * src_width + 4;
    size_t size = ((src_width + 7) & ~(size_t)7) * src_height;
    size_t step = HSW * src_height;

    size_t offs = 0;
    _Alignas(16) int16_t buf[2 * HSW];
    int16_t *ptr = buf + HSW;
    for (size_t x = HSW; x < dst_width; x += 2 * HSW) {
        for (size_t y = 0; y < src_height; y++) {
            hcopy_line(ptr - 1 * HSW, src, offs - 1 * step, size);
            hcopy_line(ptr - 0 * HSW, src, offs - 0 * step, size);
            // k=0..3 -> dst[0..7]; k=4..7 -> next[2k..2k+1] with next=dst+step-8,
            // i.e. dst[step .. step+7]
            hexpand_block(dst, dst + step, ptr);
            dst  += HSW;
            offs += HSW;
        }
        dst += step;
    }
    if ((dst_width - 1) & HSW)
        return;

    for (size_t y = 0; y < src_height; y++) {
        hcopy_line(ptr - 1 * HSW, src, offs - 1 * step, size);
        hcopy_line(ptr - 0 * HSW, src, offs - 0 * step, size);
        // only k=0..3 -> dst[0..7]
        const v128_t one = wasm_i16x8_splat(1);
        v128_t va = wasm_v128_load(ptr - 2);
        v128_t vb = wasm_v128_load(ptr);
        v128_t p1 = va;
        v128_t z0 = wasm_i16x8_shuffle(va, vb, 1, 2, 3, 4, 5, 6, 7, 14);
        v128_t n1 = vb;
        v128_t r  = wasm_u16x8_shr(wasm_i16x8_add(wasm_u16x8_shr(wasm_i16x8_add(p1, n1), 1), z0), 1);
        v128_t rp = wasm_u16x8_shr(wasm_i16x8_add(wasm_i16x8_add(wasm_u16x8_shr(wasm_i16x8_add(r, p1), 1), z0), one), 1);
        v128_t rn = wasm_u16x8_shr(wasm_i16x8_add(wasm_i16x8_add(wasm_u16x8_shr(wasm_i16x8_add(r, n1), 1), z0), one), 1);
        v128_t lo = wasm_i16x8_shuffle(rp, rn, 0, 8, 1, 9, 2, 10, 3, 11);
        wasm_v128_store(dst, lo);
        dst  += HSW;
        offs += HSW;
    }
}

static inline void blur_horz_wasm(int16_t *restrict dst, const int16_t *restrict src,
                                  size_t src_width, size_t src_height,
                                  const int16_t *restrict param, const int n)
{
    size_t dst_width = src_width + 2 * n;
    size_t size = ((src_width + 7) & ~(size_t)7) * src_height;
    size_t step = HSW * src_height;

    size_t offs = 0;
    _Alignas(16) int16_t buf[3 * HSW];
    int16_t *ptr = buf + 2 * HSW;
    for (size_t x = 0; x < dst_width; x += HSW) {
        for (size_t y = 0; y < src_height; y++) {
            for (int i = -((2 * n + HSW - 1) / HSW); i <= 0; i++)
                hcopy_line(ptr + i * HSW, src, offs + i * step, size);

            v128_t acc_lo = wasm_i32x4_splat(0x8000);
            v128_t acc_hi = acc_lo;
            v128_t center = wasm_v128_load(ptr - n);
            for (int i = n; i > 0; i--) {
                v128_t w1 = wasm_v128_load(ptr - n - i);
                v128_t w2 = wasm_v128_load(ptr - n + i);
                v128_t pv = wasm_i16x8_splat(param[i - 1]);
                // wrapping i16 sub matches the scalar (int16_t)(ptr[..] - ptr[k-n])
                v128_t d1 = wasm_i16x8_sub(w1, center);
                v128_t d2 = wasm_i16x8_sub(w2, center);
                acc_lo = wasm_i32x4_add(acc_lo, wasm_i32x4_add(
                             wasm_i32x4_extmul_low_i16x8(d1, pv),
                             wasm_i32x4_extmul_low_i16x8(d2, pv)));
                acc_hi = wasm_i32x4_add(acc_hi, wasm_i32x4_add(
                             wasm_i32x4_extmul_high_i16x8(d1, pv),
                             wasm_i32x4_extmul_high_i16x8(d2, pv)));
            }
            // dst[k] = ptr[k-n] + (acc[k] >> 16)
            v128_t lo = wasm_i32x4_add(wasm_i32x4_extend_low_i16x8(center), wasm_i32x4_shr(acc_lo, 16));
            v128_t hi = wasm_i32x4_add(wasm_i32x4_extend_high_i16x8(center), wasm_i32x4_shr(acc_hi, 16));
            wasm_v128_store(dst, wasm_i16x8_narrow_i32x4(lo, hi));

            dst  += HSW;
            offs += HSW;
        }
    }
}

void ass_blur4_horz16_wasm(int16_t *restrict dst, const int16_t *restrict src,
                           size_t w, size_t h, const int16_t *restrict param)
{ blur_horz_wasm(dst, src, w, h, param, 4); }
void ass_blur5_horz16_wasm(int16_t *restrict dst, const int16_t *restrict src,
                           size_t w, size_t h, const int16_t *restrict param)
{ blur_horz_wasm(dst, src, w, h, param, 5); }
void ass_blur6_horz16_wasm(int16_t *restrict dst, const int16_t *restrict src,
                           size_t w, size_t h, const int16_t *restrict param)
{ blur_horz_wasm(dst, src, w, h, param, 6); }
void ass_blur7_horz16_wasm(int16_t *restrict dst, const int16_t *restrict src,
                           size_t w, size_t h, const int16_t *restrict param)
{ blur_horz_wasm(dst, src, w, h, param, 7); }
void ass_blur8_horz16_wasm(int16_t *restrict dst, const int16_t *restrict src,
                           size_t w, size_t h, const int16_t *restrict param)
{ blur_horz_wasm(dst, src, w, h, param, 8); }
#undef HSW

/*
 * Rasterizer tile32 kernels (TILE_ORDER=5, selected by ASS_FLAG_LARGE_TILES).
 * Bit-exact to c/rasterizer_template.h @ TILE_SIZE==32; uses struct segment /
 * SEGFLAG_* from ass_rasterizer.h. RESCALE/FULL_VALUE are redefined locally
 * (the template #undef's them) under RK32_ names.
 */
#ifndef FFMAX
#define FFMAX(a,b) ((a) > (b) ? (a) : (b))
#endif
#ifndef FFMIN
#define FFMIN(a,b) ((a) > (b) ? (b) : (a))
#endif
#define RK32_FULL_VALUE  (1 << (14 - 5))
#define RK32_RESCALE_AB(ab, scale) \
    (((ab) * (int64_t)(scale) + ((int64_t)1 << (45 + 5))) >> (46 + 5))
#define RK32_RESCALE_C(c, scale) \
    (((int32_t)((c) >> (7 + 5)) * (int64_t)(scale) + ((int64_t)1 << 44)) >> 45)

// (c - va[x]) * w >> 16. The scalar promotes (c - va[x]) to 32-bit before the
// multiply (no int16 truncation), so the subtraction must be done in i32x4.
static inline v128_t rk32_cw(v128_t cv, v128_t vav, int32_t w) {
    v128_t clo = wasm_i32x4_extend_low_i16x8(cv);
    v128_t chi = wasm_i32x4_extend_high_i16x8(cv);
    v128_t vlo = wasm_i32x4_extend_low_i16x8(vav);
    v128_t vhi = wasm_i32x4_extend_high_i16x8(vav);
    v128_t wv = wasm_i32x4_splat(w);
    v128_t lo = wasm_i32x4_sub(clo, vlo);
    v128_t hi = wasm_i32x4_sub(chi, vhi);
    lo = wasm_i32x4_shr(wasm_i32x4_mul(lo, wv), 16);
    hi = wasm_i32x4_shr(wasm_i32x4_mul(hi, wv), 16);
    // i32 lane j sits in i16 sub-lanes 2j (low half); gather low halves back.
    return wasm_i16x8_shuffle(lo, hi, 0, 2, 4, 6, 8, 10, 12, 14);
}

static inline void rk32_update_border_line(int16_t res[32], int16_t abs_a, const int16_t va[32],
                                           int16_t b, int16_t abs_b, int16_t c, int up, int dn) {
    int16_t size = dn - up;
    int16_t w = RK32_FULL_VALUE + (size << (8 - 5)) - abs_a;
    w = FFMIN(w, RK32_FULL_VALUE) << (2 * 5 - 5);
    int16_t dc_b = abs_b * (int32_t) size >> 6;
    int16_t dc = (FFMIN(abs_a, dc_b) + 2) >> 2;
    int16_t base = (int32_t) b * (int16_t)(up + dn) >> 7;
    int16_t offs1 = size - ((base + dc) * (int32_t) w >> 16);
    int16_t offs2 = size - ((base - dc) * (int32_t) w >> 16);
    size <<= 1;
    v128_t cv = wasm_i16x8_splat(c);
    v128_t o1 = wasm_i16x8_splat(offs1);
    v128_t o2 = wasm_i16x8_splat(offs2);
    v128_t zero = wasm_i16x8_splat(0);
    v128_t sizev = wasm_i16x8_splat(size);
    for (int x = 0; x < 32; x += 8) {
        v128_t vav = wasm_v128_load(va + x);
        v128_t cw = rk32_cw(cv, vav, w);
        v128_t c1 = wasm_i16x8_add(cw, o1);
        v128_t c2 = wasm_i16x8_add(cw, o2);
        c1 = wasm_i16x8_min(wasm_i16x8_max(c1, zero), sizev);
        c2 = wasm_i16x8_min(wasm_i16x8_max(c2, zero), sizev);
        v128_t sum = wasm_i16x8_add(c1, c2);
        v128_t r = wasm_v128_load(res + x);
        wasm_v128_store(res + x, wasm_i16x8_add(r, sum));
    }
}

void ass_fill_generic_tile32_wasm(uint8_t *restrict buf, ptrdiff_t stride,
                                  const struct segment *restrict line, size_t n_lines, int winding) {
    _Alignas(16) int16_t res[32][32] = {0};
    int16_t delta[34] = {0};
    const struct segment *end = line + n_lines;
    for (; line != end; ++line) {
        int16_t up_delta = line->flags & SEGFLAG_DN ? 4 : 0;
        int16_t dn_delta = up_delta;
        if (!line->x_min && (line->flags & SEGFLAG_EXACT_LEFT)) dn_delta ^= 4;
        if (line->flags & SEGFLAG_UL_DR) { int16_t t = up_delta; up_delta = dn_delta; dn_delta = t; }
        int up = line->y_min >> 6, dn = line->y_max >> 6;
        int16_t up_pos = line->y_min & 63; int16_t up_delta1 = up_delta * up_pos;
        int16_t dn_pos = line->y_max & 63; int16_t dn_delta1 = dn_delta * dn_pos;
        delta[up + 1] -= up_delta1; delta[up] -= (up_delta << 6) - up_delta1;
        delta[dn + 1] += dn_delta1; delta[dn] += (dn_delta << 6) - dn_delta1;
        if (line->y_min == line->y_max) continue;
        int16_t a = RK32_RESCALE_AB(line->a, line->scale);
        int16_t b = RK32_RESCALE_AB(line->b, line->scale);
        int16_t c = RK32_RESCALE_C(line->c, line->scale) - (a >> 1) - b * up;
        _Alignas(16) int16_t va[32];
        for (int x = 0; x < 32; x++) va[x] = a * x;
        int16_t abs_a = a < 0 ? -a : a; int16_t abs_b = b < 0 ? -b : b;
        int16_t dc = (FFMIN(abs_a, abs_b) + 2) >> 2;
        int16_t base = RK32_FULL_VALUE / 2 - (b >> 1);
        int16_t dc1 = base + dc; int16_t dc2 = base - dc;
        if (up_pos) {
            if (dn == up) { rk32_update_border_line(res[up], abs_a, va, b, abs_b, c, up_pos, dn_pos); continue; }
            rk32_update_border_line(res[up], abs_a, va, b, abs_b, c, up_pos, 64); up++; c -= b;
        }
        {
            v128_t dc1v = wasm_i16x8_splat(dc1);
            v128_t dc2v = wasm_i16x8_splat(dc2);
            v128_t zero = wasm_i16x8_splat(0);
            v128_t fullv = wasm_i16x8_splat(RK32_FULL_VALUE);
            for (int y = up; y < dn; y++) {
                v128_t cv = wasm_i16x8_splat(c);
                for (int x = 0; x < 32; x += 8) {
                    v128_t vav = wasm_v128_load(va + x);
                    // c - va[x] truncates to int16 here, matching the int16
                    // assignment to c1/c2 in the scalar (mod 2^16); the +dc add
                    // also wraps mod 2^16, identical to the scalar.
                    v128_t cmv = wasm_i16x8_sub(cv, vav);
                    v128_t c1 = wasm_i16x8_add(cmv, dc1v);
                    v128_t c2 = wasm_i16x8_add(cmv, dc2v);
                    c1 = wasm_i16x8_min(wasm_i16x8_max(c1, zero), fullv);
                    c2 = wasm_i16x8_min(wasm_i16x8_max(c2, zero), fullv);
                    v128_t sum = wasm_i16x8_shr(wasm_i16x8_add(c1, c2), (7 - 5));
                    v128_t r = wasm_v128_load(res[y] + x);
                    wasm_v128_store(res[y] + x, wasm_i16x8_add(r, sum));
                }
                c -= b;
            }
        }
        if (dn_pos) rk32_update_border_line(res[dn], abs_a, va, b, abs_b, c, 0, dn_pos);
    }
    int16_t cur = 256 * (int8_t) winding;
    for (int y = 0; y < 32; y++) {
        cur += delta[y];
        v128_t curv = wasm_i16x8_splat(cur);
        v128_t c255 = wasm_i16x8_splat(255);
        v128_t zero = wasm_i16x8_splat(0);
        for (int x = 0; x < 32; x += 16) {
            v128_t r0 = wasm_v128_load(res[y] + x);
            v128_t r1 = wasm_v128_load(res[y] + x + 8);
            v128_t v0 = wasm_i16x8_add(r0, curv);
            v128_t v1 = wasm_i16x8_add(r1, curv);
            // abs(val) == max(val, -val); -val via 0 - val (int16 wrap matches scalar)
            v0 = wasm_i16x8_max(v0, wasm_i16x8_sub(zero, v0));
            v1 = wasm_i16x8_max(v1, wasm_i16x8_sub(zero, v1));
            v0 = wasm_i16x8_min(v0, c255);
            v1 = wasm_i16x8_min(v1, c255);
            wasm_v128_store(buf + x, wasm_u8x16_narrow_i16x8(v0, v1));
        }
        buf += stride;
    }
}

void ass_merge_tile32_wasm(uint8_t *restrict buf, ptrdiff_t stride, const uint8_t *restrict tile) {
    for (int y = 0; y < 32; y++) {
        wasm_v128_store(buf, wasm_u8x16_max(wasm_v128_load(buf), wasm_v128_load(tile)));
        wasm_v128_store(buf + 16, wasm_u8x16_max(wasm_v128_load(buf + 16), wasm_v128_load(tile + 16)));
        buf += stride; tile += 32;
    }
}

void ass_fill_solid_tile32_wasm(uint8_t *buf, ptrdiff_t stride, int set) {
    v128_t v = wasm_u8x16_splat(set ? 255 : 0);
    for (int y = 0; y < 32; y++) {
        wasm_v128_store(buf, v);
        wasm_v128_store(buf + 16, v);
        buf += stride;
    }
}

void ass_fill_halfplane_tile32_wasm(uint8_t *buf, ptrdiff_t stride,
                                    int32_t a, int32_t b, int64_t c, int32_t scale) {
    int16_t aa = RK32_RESCALE_AB(a, scale), bb = RK32_RESCALE_AB(b, scale);
    int16_t cc = RK32_RESCALE_C(c, scale) + RK32_FULL_VALUE / 2 - ((aa + bb) >> 1);
    int16_t abs_a = aa < 0 ? -aa : aa; int16_t abs_b = bb < 0 ? -bb : bb;
    int16_t delta = (FFMIN(abs_a, abs_b) + 2) >> 2;
    _Alignas(16) int16_t va1[32], va2[32];
    for (int x = 0; x < 32; x++) { va1[x] = aa * x - delta; va2[x] = aa * x + delta; }
    v128_t zero = wasm_i16x8_splat(0);
    v128_t fullv = wasm_i16x8_splat(RK32_FULL_VALUE);
    v128_t c255 = wasm_i16x8_splat(255);
    for (int y = 0; y < 32; y++) {
        v128_t ccv = wasm_i16x8_splat(cc);
        for (int x = 0; x < 32; x += 16) {
            v128_t c1_0 = wasm_i16x8_sub(ccv, wasm_v128_load(va1 + x));
            v128_t c1_1 = wasm_i16x8_sub(ccv, wasm_v128_load(va1 + x + 8));
            v128_t c2_0 = wasm_i16x8_sub(ccv, wasm_v128_load(va2 + x));
            v128_t c2_1 = wasm_i16x8_sub(ccv, wasm_v128_load(va2 + x + 8));
            c1_0 = wasm_i16x8_min(wasm_i16x8_max(c1_0, zero), fullv);
            c1_1 = wasm_i16x8_min(wasm_i16x8_max(c1_1, zero), fullv);
            c2_0 = wasm_i16x8_min(wasm_i16x8_max(c2_0, zero), fullv);
            c2_1 = wasm_i16x8_min(wasm_i16x8_max(c2_1, zero), fullv);
            v128_t r0 = wasm_i16x8_min(wasm_i16x8_shr(wasm_i16x8_add(c1_0, c2_0), (7 - 5)), c255);
            v128_t r1 = wasm_i16x8_min(wasm_i16x8_shr(wasm_i16x8_add(c1_1, c2_1), (7 - 5)), c255);
            wasm_v128_store(buf + x, wasm_u8x16_narrow_i16x8(r0, r1));
        }
        buf += stride; cc -= bb;
    }
}

/*
 * be_blur (VSFilter [1,2,1] box blur). The horizontal [1,2,1] is a 2-tap done
 * with shifted loads; the vertical pass carries per-column state in
 * col_pix_buf/col_sum_buf and is vectorized across x.
 */

/* Horizontal [1,2,1] for 8 interior columns at index x (requires 1<=x and
 * x+7<=width-2, so all src loads stay in [0,width-1]):
 *   cp[i] = src[i-1] + 2*src[i] + src[i+1]   */
static inline v128_t ass_be_horiz8_wasm(const uint8_t *restrict src, size_t x)
{
    v128_t l = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(src + x - 1));
    v128_t c = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(src + x));
    v128_t r = wasm_u16x8_extend_low_u8x16(wasm_v128_load64_zero(src + x + 1));
    return wasm_i16x8_add(wasm_i16x8_add(l, r), wasm_i16x8_shl(c, 1));
}

/* Vertical fold for 8 columns at index x using horizontal result cpv:
 *   col_sum = col_pix_buf[x] + cp;   col_pix_buf[x] = cp;
 *   sumout  = col_sum_buf[x] + col_sum; col_sum_buf[x] = col_sum;
 *   dst[x]  = sumout >> 4;   */
static inline void ass_be_vfold8_wasm(uint8_t *restrict dst,
                                      uint16_t *restrict col_pix_buf,
                                      uint16_t *restrict col_sum_buf,
                                      v128_t cpv, size_t x)
{
    v128_t cpb = wasm_v128_load(col_pix_buf + x);
    v128_t csb = wasm_v128_load(col_sum_buf + x);
    v128_t col_sum = wasm_i16x8_add(cpb, cpv);
    v128_t sumout  = wasm_i16x8_add(csb, col_sum);
    wasm_v128_store(col_pix_buf + x, cpv);
    wasm_v128_store(col_sum_buf + x, col_sum);
    v128_t outsh = wasm_u16x8_shr(sumout, 4);
    wasm_v128_store64_lane(dst + x, wasm_u8x16_narrow_i16x8(outsh, outsh), 0);
}

void ass_be_blur_wasm(uint8_t *restrict buf, ptrdiff_t stride,
                      size_t width, size_t height, uint16_t *restrict tmp)
{
    ASSUME(!((uintptr_t) buf % ALIGNMENT) && !(stride % ALIGNMENT));
    ASSUME(!((uintptr_t) tmp % ALIGNMENT));
    ASSUME(width > 1 && height > 1);

    uint16_t *col_pix_buf = tmp;
    uint16_t *col_sum_buf = tmp + stride;

    /* First row: col_pix_buf[x] = col_sum_buf[x] = horiz(buf)[x] */
    {
        const uint8_t *src = buf;
        size_t x = 1;
        for (; x + 8 <= width - 1; x += 8) {            /* interior, x+7 <= width-2 */
            v128_t cpv = ass_be_horiz8_wasm(src, x);
            wasm_v128_store(col_pix_buf + x, cpv);
            wasm_v128_store(col_sum_buf + x, cpv);
        }
        for (; x + 1 < width; x++) {                    /* x <= width-2 */
            uint16_t v = (uint16_t)(src[x - 1] + 2 * src[x] + src[x + 1]);
            col_pix_buf[x] = col_sum_buf[x] = v;
        }
        uint16_t v0 = (uint16_t)(2 * src[0] + src[1]);
        uint16_t vN = (uint16_t)(src[width - 2] + 2 * src[width - 1]);
        col_pix_buf[0] = col_sum_buf[0] = v0;
        col_pix_buf[width - 1] = col_sum_buf[width - 1] = vN;
    }

    /* Main rows */
    for (size_t y = 1; y < height; y++) {
        uint8_t *dst = buf;
        buf += stride;
        const uint8_t *src = buf;

        size_t x = 1;
        for (; x + 8 <= width - 1; x += 8) {            /* interior columns */
            v128_t cpv = ass_be_horiz8_wasm(src, x);
            ass_be_vfold8_wasm(dst, col_pix_buf, col_sum_buf, cpv, x);
        }
        for (; x + 1 < width; x++) {                    /* scalar interior tail */
            uint16_t cp = (uint16_t)(src[x - 1] + 2 * src[x] + src[x + 1]);
            uint16_t col_sum = (uint16_t)(col_pix_buf[x] + cp);
            col_pix_buf[x] = cp;
            uint16_t sumout = (uint16_t)(col_sum_buf[x] + col_sum);
            col_sum_buf[x] = col_sum;
            dst[x] = (uint8_t)(sumout >> 4);
        }
        {   /* edge column 0 */
            uint16_t cp = (uint16_t)(2 * src[0] + src[1]);
            uint16_t col_sum = (uint16_t)(col_pix_buf[0] + cp);
            col_pix_buf[0] = cp;
            uint16_t sumout = (uint16_t)(col_sum_buf[0] + col_sum);
            col_sum_buf[0] = col_sum;
            dst[0] = (uint8_t)(sumout >> 4);
        }
        {   /* edge column width-1 */
            size_t w1 = width - 1;
            uint16_t cp = (uint16_t)(src[width - 2] + 2 * src[w1]);
            uint16_t col_sum = (uint16_t)(col_pix_buf[w1] + cp);
            col_pix_buf[w1] = cp;
            uint16_t sumout = (uint16_t)(col_sum_buf[w1] + col_sum);
            col_sum_buf[w1] = col_sum;
            dst[w1] = (uint8_t)(sumout >> 4);
        }
    }

    /* Final flush: buf points at the last row */
    {
        size_t x = 0;
        for (; x + 8 <= width; x += 8) {
            v128_t csb = wasm_v128_load(col_sum_buf + x);
            v128_t cpb = wasm_v128_load(col_pix_buf + x);
            v128_t s = wasm_u16x8_shr(wasm_i16x8_add(csb, cpb), 4);
            wasm_v128_store64_lane(buf + x, wasm_u8x16_narrow_i16x8(s, s), 0);
        }
        for (; x < width; x++)
            buf[x] = (uint8_t)((uint16_t)(col_sum_buf[x] + col_pix_buf[x]) >> 4);
    }
}
#endif


#define RASTERIZER_PROTOTYPES(tile_size, suffix) \
    FillSolidTileFunc     ass_fill_solid_tile     ## tile_size ## _ ## suffix; \
    FillHalfplaneTileFunc ass_fill_halfplane_tile ## tile_size ## _ ## suffix; \
    FillGenericTileFunc   ass_fill_generic_tile   ## tile_size ## _ ## suffix; \
    MergeTileFunc         ass_merge_tile          ## tile_size ## _ ## suffix;

#define RASTERIZER_FUNCTION(name, suffix) \
    engine.name = mask & ASS_FLAG_LARGE_TILES ? \
        ass_ ## name ## _tile32_ ## suffix : \
        ass_ ## name ## _tile16_ ## suffix;

#define RASTERIZER_FUNCTIONS(suffix) \
    RASTERIZER_FUNCTION(fill_solid,     suffix) \
    RASTERIZER_FUNCTION(fill_halfplane, suffix) \
    RASTERIZER_FUNCTION(fill_generic,   suffix) \
    RASTERIZER_FUNCTION(merge,          suffix)


#define GENERIC_PROTOTYPES(suffix) \
    BitmapBlendFunc ass_add_bitmaps_  ## suffix; \
    BitmapBlendFunc ass_imul_bitmaps_ ## suffix; \
    BitmapMulFunc   ass_mul_bitmaps_  ## suffix; \
    BeBlurFunc      ass_be_blur_      ## suffix;

#define GENERIC_FUNCTION(name, suffix) \
    engine.name = ass_ ## name ## _ ## suffix;

#define GENERIC_FUNCTIONS(suffix) \
    GENERIC_FUNCTION(add_bitmaps,  suffix) \
    GENERIC_FUNCTION(imul_bitmaps, suffix) \
    GENERIC_FUNCTION(mul_bitmaps,  suffix) \
    GENERIC_FUNCTION(be_blur,      suffix)


#define PARAM_BLUR_SET(suffix) \
    ass_blur4_ ## suffix, \
    ass_blur5_ ## suffix, \
    ass_blur6_ ## suffix, \
    ass_blur7_ ## suffix, \
    ass_blur8_ ## suffix

#define BLUR_PROTOTYPES(stripe_width, suffix) \
    Convert8to16Func ass_stripe_unpack  ## stripe_width ## _ ## suffix; \
    Convert16to8Func ass_stripe_pack    ## stripe_width ## _ ## suffix; \
    FilterFunc       ass_shrink_horz    ## stripe_width ## _ ## suffix; \
    FilterFunc       ass_shrink_vert    ## stripe_width ## _ ## suffix; \
    FilterFunc       ass_expand_horz    ## stripe_width ## _ ## suffix; \
    FilterFunc       ass_expand_vert    ## stripe_width ## _ ## suffix; \
    ParamFilterFunc PARAM_BLUR_SET(horz ## stripe_width ## _ ## suffix); \
    ParamFilterFunc PARAM_BLUR_SET(vert ## stripe_width ## _ ## suffix);

#define BLUR_FUNCTION(name, alignment, suffix) \
    engine.name = ass_ ## name ## alignment ## _ ## suffix;

#define PARAM_BLUR_FUNCTION(dir, alignment, suffix) \
    engine.blur_ ## dir[0] = ass_blur4_ ## dir ## alignment ## _ ## suffix; \
    engine.blur_ ## dir[1] = ass_blur5_ ## dir ## alignment ## _ ## suffix; \
    engine.blur_ ## dir[2] = ass_blur6_ ## dir ## alignment ## _ ## suffix; \
    engine.blur_ ## dir[3] = ass_blur7_ ## dir ## alignment ## _ ## suffix; \
    engine.blur_ ## dir[4] = ass_blur8_ ## dir ## alignment ## _ ## suffix;

#define BLUR_FUNCTIONS(align_order_, alignment, suffix) \
    BLUR_FUNCTION(stripe_unpack, alignment, suffix) \
    BLUR_FUNCTION(stripe_pack,   alignment, suffix) \
    BLUR_FUNCTION(shrink_horz,   alignment, suffix) \
    BLUR_FUNCTION(shrink_vert,   alignment, suffix) \
    BLUR_FUNCTION(expand_horz,   alignment, suffix) \
    BLUR_FUNCTION(expand_vert,   alignment, suffix) \
    PARAM_BLUR_FUNCTION(horz, alignment, suffix) \
    PARAM_BLUR_FUNCTION(vert, alignment, suffix) \
    engine.align_order = align_order_;


#define ALL_PROTOTYPES(alignment, suffix) \
    RASTERIZER_PROTOTYPES(16, suffix) \
    RASTERIZER_PROTOTYPES(32, suffix) \
    GENERIC_PROTOTYPES(suffix) \
    BLUR_PROTOTYPES(alignment, suffix)

#define ALL_FUNCTIONS(align_order_, alignment, suffix) \
    RASTERIZER_FUNCTIONS(suffix) \
    GENERIC_FUNCTIONS(suffix) \
    BLUR_FUNCTIONS(align_order_, alignment, suffix)


unsigned ass_get_cpu_flags(unsigned mask)
{
    unsigned flags = ASS_CPU_FLAG_NONE;

#if CONFIG_ASM && ARCH_X86

    if (!ass_has_cpuid())
        return flags & mask;

    uint32_t eax = 0, ebx, ecx, edx;
    ass_get_cpuid(&eax, &ebx, &ecx, &edx);
    uint32_t max_leaf = eax;

    bool avx = false;
    if (max_leaf >= 1) {
        eax = 1;
        ass_get_cpuid(&eax, &ebx, &ecx, &edx);
        if (edx & (1 << 26)) {  // SSE2
            flags |= ASS_CPU_FLAG_X86_SSE2;
            if (ecx & (1 << 0) &&  // SSE3
                ecx & (1 << 9))    // SSSE3
                    flags |= ASS_CPU_FLAG_X86_SSSE3;
        }

        if (ecx & (1 << 27) &&  // OSXSAVE
            ecx & (1 << 28)) {  // AVX
            uint32_t xcr0l, xcr0h;
            ass_get_xgetbv(0, &xcr0l, &xcr0h);
            if (xcr0l & (1 << 1) &&  // XSAVE for XMM
                xcr0l & (1 << 2))    // XSAVE for YMM
                    avx = true;
        }
    }

    if (max_leaf >= 7) {
        eax = 7;
        ass_get_cpuid(&eax, &ebx, &ecx, &edx);
        if (avx && ebx & (1 << 5))  // AVX2
            flags |= ASS_CPU_FLAG_X86_AVX2;
    }

#endif

#if ARCH_AARCH64
    flags = ASS_CPU_FLAG_ARM_NEON;
#endif

    return flags & mask;
}

BitmapEngine ass_bitmap_engine_init(unsigned mask)
{
    ALL_PROTOTYPES(16, c)
    BLUR_PROTOTYPES(32, c)
    BitmapEngine engine = {0};
    engine.tile_order = mask & ASS_FLAG_LARGE_TILES ? 5 : 4;

#if CONFIG_ASM
    unsigned flags = ass_get_cpu_flags(mask);
#if ARCH_X86
    if (flags & ASS_CPU_FLAG_X86_AVX2) {
        ALL_PROTOTYPES(32, avx2)
        ALL_FUNCTIONS(5, 32, avx2)
        return engine;
    } else if (flags & ASS_CPU_FLAG_X86_SSE2) {
        ALL_PROTOTYPES(16, sse2)
        ALL_FUNCTIONS(4, 16, sse2)
        if (flags & ASS_CPU_FLAG_X86_SSSE3) {
            ALL_PROTOTYPES(16, ssse3)
            RASTERIZER_FUNCTION(fill_generic, ssse3)
            GENERIC_FUNCTION(be_blur, ssse3)
            BLUR_FUNCTION(shrink_horz, 16, ssse3)
            BLUR_FUNCTION(expand_horz, 16, ssse3)
            PARAM_BLUR_FUNCTION(horz, 16, ssse3)
        }
        return engine;
    }
#elif ARCH_AARCH64
    if (flags & ASS_CPU_FLAG_ARM_NEON) {
        ALL_PROTOTYPES(16, neon)
        ALL_FUNCTIONS(4, 16, neon)
        return engine;
    }
#endif
#endif

    ALL_FUNCTIONS(4, 16, c)
    if (mask & ASS_FLAG_WIDE_STRIPE) {
        BLUR_FUNCTIONS(5, 32, c)
    }

#if defined(__wasm_simd128__)
    engine.add_bitmaps  = ass_add_bitmaps_wasm;
    engine.imul_bitmaps = ass_imul_bitmaps_wasm;
    engine.mul_bitmaps  = ass_mul_bitmaps_wasm;
    engine.be_blur      = ass_be_blur_wasm;
    if (!(mask & ASS_FLAG_WIDE_STRIPE)) {
        engine.stripe_unpack = ass_stripe_unpack16_wasm;
        engine.stripe_pack   = ass_stripe_pack16_wasm;
        engine.shrink_horz   = ass_shrink_horz16_wasm;
        engine.shrink_vert   = ass_shrink_vert16_wasm;
        engine.expand_horz   = ass_expand_horz16_wasm;
        engine.expand_vert   = ass_expand_vert16_wasm;
        engine.blur_horz[0]  = ass_blur4_horz16_wasm;
        engine.blur_horz[1]  = ass_blur5_horz16_wasm;
        engine.blur_horz[2]  = ass_blur6_horz16_wasm;
        engine.blur_horz[3]  = ass_blur7_horz16_wasm;
        engine.blur_horz[4]  = ass_blur8_horz16_wasm;
        engine.blur_vert[0]  = ass_blur4_vert16_wasm;
        engine.blur_vert[1]  = ass_blur5_vert16_wasm;
        engine.blur_vert[2]  = ass_blur6_vert16_wasm;
        engine.blur_vert[3]  = ass_blur7_vert16_wasm;
        engine.blur_vert[4]  = ass_blur8_vert16_wasm;
    }
    if (mask & ASS_FLAG_LARGE_TILES) {
        engine.fill_solid     = ass_fill_solid_tile32_wasm;
        engine.fill_halfplane = ass_fill_halfplane_tile32_wasm;
        engine.fill_generic   = ass_fill_generic_tile32_wasm;
        engine.merge          = ass_merge_tile32_wasm;
    }
#endif

    return engine;
}
