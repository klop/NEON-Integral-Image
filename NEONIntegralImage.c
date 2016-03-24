//
//  NEONIntegralImage.c
//  NEONIntegralImage
//
//  Created by Dara on 23/03/2016.
//  Copyright © 2016 Dara. All rights reserved.
//

#include <stdio.h>
#include <string.h>
#include <arm_neon.h>

void neon_integral_image(const uint8_t *sourceImage, uint32_t *integralImage,
                         size_t width, size_t height)
{
    // integral images add an extra row and column of 0s to the image
    size_t integralImageWidth = width + 1;

    // 0 out the first row
    memset(integralImage, 0, sizeof(integralImage[0]) * integralImageWidth);

    // pointer to the start of the integral image, skipping past the first row and column
    uint32_t *integralImageStart = integralImage + integralImageWidth + 1;

    //  used to carry over the last prefix sum of a row segment to the next segment
    uint32_t prefixSumLastElement = 0;

    // a vector used later for shifting bytes and replacing with 0 using vector extraction
    uint16x8_t zeroVec = vdupq_n_u16(0);

    // prefix sum for rows
    for (size_t i = 0; i < height; ++i) {

        // vector to carry over last prefix sum value to next chunk
        uint32x4_t carry = vdupq_n_u32(0);

        // get offset to start of row, taking into account that we have one more column and row than the source image
        size_t sourceRowOffset = i * width;

        // from the integral image start, this gives us an offset to the beginning of the next row, skipping the 0 column
        size_t integralRowOffset = i * integralImageWidth;

        // 0 out the start of every row, starting from the 2nd
        integralImage[integralImageWidth + integralRowOffset] = 0;

        // count how many bytes we've passed over
        size_t j = 0;

        // iterate over the row in 16 byte chunks
        for (; j + 16 < width; j += 16) {

            // load 16 bytes/ints from given offset
            uint8x16_t elements = vld1q_u8(sourceImage + sourceRowOffset + j);

            // take lower 8 8-bit ints
            uint8x8_t lowElements8 = vget_low_u8(elements);

            // convert them to 16-bit ints
            uint16x8_t lowElements16 = vmovl_u8(lowElements8);

            // take upper 8 8-bit ints
            uint8x8_t highElements8 = vget_high_u8(elements);

            // convert them to 16-bit ints
            uint16x8_t highElements16 = vmovl_u8(highElements8);

            // add lowElements16 to lowElements16 shifted 2 bytes (1 element) to the right
            lowElements16 = vaddq_u16(lowElements16, vextq_u16(zeroVec, lowElements16, 7));

            // add result to result shifted 4 bytes (2 elements) to the right
            lowElements16 = vaddq_u16(lowElements16, vextq_u16(zeroVec, lowElements16, 6));

            // add result to result shifted 8 bytes (4 elements) to the right, we now have the prefix sums for this section
            lowElements16 = vaddq_u16(lowElements16, vextq_u16(zeroVec, lowElements16, 4));

            // do the same 3 steps above for highElements16
            highElements16 = vaddq_u16(highElements16, vextq_u16(zeroVec, highElements16, 7));
            highElements16 = vaddq_u16(highElements16, vextq_u16(zeroVec, highElements16, 6));
            highElements16 = vaddq_u16(highElements16, vextq_u16(zeroVec, highElements16, 4));

            // take lower 4 16-bit ints of lowElements16, convert to 32 bit, and add to carry (carry is 0s for the first 8 pixels in each row)
            uint32x4_t lowElementsOfLowPrefix32 = vaddq_u32(vmovl_u16(vget_low_u16(lowElements16)), carry);

            // store lower 4 32-bit ints at appropriate offset: (X|O|O|O)
            vst1q_u32(integralImageStart + integralRowOffset + j, lowElementsOfLowPrefix32);

            // take upper 4 16-bit ints of lowElements16, convert to 32 bit, and add to carry
            uint32x4_t highElementsOfLowPrefix32 = vaddq_u32(vmovl_u16(vget_high_u16(lowElements16)), carry);

            // store upper 4 32-bit ints at appropriate offset: (O|X|O|O)
            vst1q_u32(integralImageStart + integralRowOffset + j + 4, highElementsOfLowPrefix32);

            // take the last prefix sum from the second 32-bit vector to be added to the next prefix sums
            prefixSumLastElement = vgetq_lane_u32(highElementsOfLowPrefix32, 3);

            // fill carry vector with 4 32-bit ints, each with the value of the last prefix sum element
            carry = vdupq_n_u32(prefixSumLastElement);

            // take lower 4 16-bit ints of lowElements16, convert to 32 bit, and add to carry (carry is 0s for the first pass of each row)
            uint32x4_t lowElementsOfHighPrefix32 = vaddq_u32(vmovl_u16(vget_low_u16(highElements16)), carry);

            // store lower 4 32-bit ints at appropriate offset: (O|O|X|O)
            vst1q_u32(integralImageStart + integralRowOffset + j + 8, lowElementsOfHighPrefix32);

            // take upper 4 16-bit ints of lowElements16, convert to 32 bit, and add to carry
            uint32x4_t highElementsOfHighPrefix32 = vaddq_u32(vmovl_u16(vget_high_u16(highElements16)), carry);

            // store upper 4 32-bit ints at appropriate offset (O|O|O|X)
            vst1q_u32(integralImageStart + integralRowOffset + j + 12, highElementsOfHighPrefix32);

            // take the last prefix sum from the second 32-bit vector to be added to the next prefix sums
            prefixSumLastElement = vgetq_lane_u32(highElementsOfHighPrefix32, 3);

            // fill carry vector with 4 32-bit ints, each with the value of the last prefix sum element
            carry = vdupq_n_u32(prefixSumLastElement);
        }

        // now handle the remainders (< 16 pixels)
        for (; j < width ; ++j) {

            // take the last prefix sum value and add the 8-bit int value from the source image at the appropriate index
            prefixSumLastElement += sourceImage[sourceRowOffset + j];

            // set the value of the integral image to the last prefix sum, at the same index
            integralImageStart[integralRowOffset + j] = prefixSumLastElement;
        }
    }

    // prefix sum for columns, using height - 1 since we're taking pairs of rows
    for (size_t i = 0; i < height - 1; ++i) {
        size_t j = 0;
        size_t integralRowOffset = i * integralImageWidth;

        for (; j + 16 < width; j += 16) {

            // load 4 32-bit ints from row i (first row)
            uint32x4_t row1Elements32 = vld1q_u32(integralImageStart + integralRowOffset + j);

            // load 4 32-bit ints from row i + 1 (second row)
            uint32x4_t row2Elements32 = vld1q_u32(integralImageStart + integralRowOffset + integralImageWidth + j);

            // add first row to second row, giving the prefix sum for the second row
            row2Elements32 = vqaddq_u32(row1Elements32, row2Elements32);

            // replace the stored second row values with the prefix sum values
            vst1q_u32(integralImageStart + integralRowOffset + integralImageWidth + j, row2Elements32);

            // do the same for the next 3 offsets (4, 8, 12), completing a 128-bit chunk
            row1Elements32 = vld1q_u32(integralImageStart + integralRowOffset + j + 4);
            row2Elements32 = vld1q_u32(integralImageStart + integralRowOffset + integralImageWidth + j + 4);
            row2Elements32 = vqaddq_u32(row1Elements32, row2Elements32);
            vst1q_u32(integralImageStart + integralRowOffset + integralImageWidth + j + 4, row2Elements32);

            row1Elements32 = vld1q_u32(integralImageStart + integralRowOffset + j + 8);
            row2Elements32 = vld1q_u32(integralImageStart + integralRowOffset + integralImageWidth + j + 8);
            row2Elements32 = vqaddq_u32(row1Elements32, row2Elements32);
            vst1q_u32(integralImageStart + integralRowOffset + integralImageWidth + j + 8, row2Elements32);

            row1Elements32 = vld1q_u32(integralImageStart + integralRowOffset + j + 12);
            row2Elements32 = vld1q_u32(integralImageStart + integralRowOffset + integralImageWidth + j + 12);
            row2Elements32 = vqaddq_u32(row1Elements32, row2Elements32);
            vst1q_u32(integralImageStart + integralRowOffset + integralImageWidth + j + 12, row2Elements32);
        }
        
        // now handle the remainders
        for (; j < width; ++j) {
            integralImageStart[integralRowOffset + integralImageWidth + j] += integralImageStart[integralRowOffset + j];
        }
    }
}
