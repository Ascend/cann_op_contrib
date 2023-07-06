/**
 * Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <gtest/gtest.h>
#include <vector>
#include "ascendc_ut_util.h"
#include "tikicpulib.h"

extern "C" void add_custom(uint8_t* x, uint8_t* y, uint8_t* z, uint8_t* tiling);

class AddTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "add test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "add test TearDown" << std::endl;
    }
};

TEST_F(AddTest, add_test_case_1) {
    size_t tilingSize = 3 * sizeof(uint32_t);
    uint8_t* tiling = (uint8_t*)tik2::GmAlloc(tilingSize);
    ReadFile(ktestcaseFilePath + "add_custom/data/tiling.bin", tilingSize, tiling, tilingSize);

    uint32_t blockDim = (*(const uint32_t*)(tiling));
    size_t inputByteSize = blockDim * 2048 * sizeof(uint16_t);
    size_t outputByteSize = blockDim * 2048 * sizeof(uint16_t);

    uint8_t* x = (uint8_t*)tik2::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)tik2::GmAlloc(inputByteSize);
    uint8_t* z = (uint8_t*)tik2::GmAlloc(outputByteSize);

    ReadFile(ktestcaseFilePath + "add_custom/data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(ktestcaseFilePath + "add_custom/data/input_y.bin", inputByteSize, y, inputByteSize);

    ICPU_RUN_KF(add_custom, blockDim, x, y, z, tiling);

    WriteFile(ktestcaseFilePath + "add_custom/data/output_z.bin", z, outputByteSize);

    size_t elementsNum = blockDim * 2048 ;
    half* golden = new half[elementsNum];
    auto goldenFilePath = ktestcaseFilePath + "add_custom/data/golden.txt";
    ReadFile(goldenFilePath, golden, elementsNum);
    bool compare = CompareResult((half*)z, golden, elementsNum);

    tik2::GmFree((void*)x);
    tik2::GmFree((void*)y);
    tik2::GmFree((void*)z);
    tik2::GmFree((void*)tiling);

    EXPECT_EQ(compare, true);
}
