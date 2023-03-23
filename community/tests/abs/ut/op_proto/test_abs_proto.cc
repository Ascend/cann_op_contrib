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
#include "abs_op.h"

class AbsTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "Abs test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Abs test TearDown" << std::endl;
  }
};

TEST_F(AbsTest, Abs_test_case_1) {
   ge::op::Abs abs_op;
   ge::TensorDesc tensorDesc;
   ge::Shape shape({2, 3, 4});
   tensorDesc.SetDataType(ge::DT_FLOAT16);
   tensorDesc.SetShape(shape);
   tensorDesc.SetOriginShape(shape);
   abs_op.UpdateInputDesc("x", tensorDesc);
   auto ret = abs_op.InferShapeAndType();
   EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
   auto output_desc = abs_op.GetOutputDescByName("y");
   EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
   std::vector<int64_t> expected_output_shape = {2, 3, 4};
   EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
