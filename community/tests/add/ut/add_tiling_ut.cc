/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "gtest/gtest.h"

class add_tiling_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "add_tiling_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "add_tiling_test TearDown" << std::endl;
  }
};

TEST_F(add_tiling_test, add_tiling_test_001) {
  int ret = 1;
  EXPECT_EQ(ret, 1);
}