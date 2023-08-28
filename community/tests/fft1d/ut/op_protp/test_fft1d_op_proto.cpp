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
#include "fft1d.h"

class FFT1DTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "fft1d test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "fft1d test TearDown" << std::endl;
  }
};

TEST_F(FFT1DTest, fft1d_test_case_1) {
   ge::op::FFT1D fft1d_op;
   auto ret = fft1d_op.InferShapeAndType();
   EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
   auto ret2 = fft1d_op.VerifyAllAttr(true);
   EXPECT_EQ(ret2, ge::GRAPH_SUCCESS);
}