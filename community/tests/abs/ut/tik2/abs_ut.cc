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
#include "tik2_ut_util.h"
using namespace std;

extern "C" void abs_tik2(uint8_t* x, uint8_t* y);

class AbsTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "add test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "add test TearDown" << std::endl;
    }
};

TEST_F(AbsTest, abs_test_case_1) {
    auto exp_out_file_path = ktestcaseFilePath + "abs/data/output_y.txt";
    size_t elements_num = 16 * 32;
    size_t inputByteSize = 16 * 32 * sizeof(uint16_t);
    size_t outputByteSize = 16 * 32 * sizeof(uint16_t);
    uint32_t blockDim = 1;
    uint8_t* x = (uint8_t*)gm_alloc(inputByteSize);
    uint8_t* y = (uint8_t*)gm_alloc(outputByteSize);
    ReadFile(ktestcaseFilePath + "abs/data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_RUN_KF(abs_tik2, blockDim, x, y); // use this macro for cpu debug
    
    half *output_exp = new half[elements_num];
    half* real_out = (half *) y;
    ReadFile(exp_out_file_path, output_exp, elements_num);
    bool compare = CompareResult(real_out, output_exp, elements_num);
    EXPECT_EQ(compare, true);
}