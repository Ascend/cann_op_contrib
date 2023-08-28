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
#include "stft.h"

class STFTTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "stft test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "stft test TearDown" << std::endl;
  }
};

TEST_F(STFTTest, stft_test_case_1) {
   ge::op::STFT stft_op;
   auto ret = stft_op.InferShapeAndType();
   EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
   auto ret2 = stft_op.VerifyAllAttr(true);
   EXPECT_EQ(ret2, ge::GRAPH_SUCCESS);
}
