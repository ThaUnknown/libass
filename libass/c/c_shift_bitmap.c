/*
 * Copyright (C) 2026 libass contributors
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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ass_utils.h"


#define ALIGNMENT  16

/*
 * Shifting moves (pixel * shift >> 6) of every pixel but the last one
 * to its right (bottom) neighbor, all amounts being taken from the
 * original pixels:
 *   out[x] = in[x] - (x < w - 1 ? in[x] * shift >> 6 : 0)
 *                  + (x > 0     ? in[x - 1] * shift >> 6 : 0)
 */

static inline uint8_t shift_part(uint8_t val, int shift)
{
    return val * shift >> 6;
}

/*
 * Process the row right to left in blocks, so that every block only reads
 * pixels that are not modified yet, and every block can be vectorized.
 */
static void shift_line_horz(uint8_t *row, size_t width, int shift)
{
    uint8_t last = row[width - 1] + shift_part(row[width - 2], shift);
    size_t x = width - 2;
    for (; x >= 16; x -= 16) {
        uint8_t *ptr = row + x - 15, res[16];
        for (int i = 0; i < 16; i++)
            res[i] = ptr[i] + shift_part(ptr[i - 1], shift) -
                shift_part(ptr[i], shift);
        memcpy(ptr, res, sizeof(res));
    }
    for (; x > 0; x--)
        row[x] += shift_part(row[x - 1], shift) - shift_part(row[x], shift);
    row[0] -= shift_part(row[0], shift);
    row[width - 1] = last;
}

// rows cannot overlap as stride >= width
static void shift_line_vert(uint8_t *restrict cur, uint8_t *restrict prev,
                            size_t width, int shift)
{
    for (size_t x = 0; x < width; x++) {
        uint8_t b = shift_part(prev[x], shift);
        prev[x] -= b;
        cur[x] += b;
    }
}

/**
 * \brief Shift bitmap rows by shift / 64 pixel to the right
 */
void ass_shift_horz_c(uint8_t *buf, ptrdiff_t stride,
                      size_t width, size_t height, int shift)
{
    ASSUME(!((uintptr_t) buf % ALIGNMENT) && !(stride % ALIGNMENT));
    ASSUME(width > 0 && height > 0);

    if (width < 2)
        return;

    uint8_t *end = buf + stride * height;
    for (uint8_t *row = buf; row < end; row += stride)
        shift_line_horz(row, width, shift);
}

/**
 * \brief Shift bitmap columns by shift / 64 pixel to the bottom
 * Rows are processed bottom to top, so that the row above is not modified yet.
 */
void ass_shift_vert_c(uint8_t *buf, ptrdiff_t stride,
                      size_t width, size_t height, int shift)
{
    ASSUME(!((uintptr_t) buf % ALIGNMENT) && !(stride % ALIGNMENT));
    ASSUME(width > 0 && height > 0);

    for (uint8_t *row = buf + stride * (height - 1); row > buf; row -= stride)
        shift_line_vert(row, row - stride, width, shift);
}
