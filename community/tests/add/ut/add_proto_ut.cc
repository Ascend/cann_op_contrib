#include "gtest/gtest.h"

class add_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "add_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "add_test TearDown" << std::endl;
  }
};

TEST_F(add_test, add_test_001) {
  int ret = 1;
  EXPECT_EQ(ret, 1);
}