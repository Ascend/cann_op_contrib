#include <gtest/gtest.h>
#include <vector>
#include "inc/sinh_op.h"

class SinhTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "sinh test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "sinh test TearDown" << std::endl;
  }
};

TEST_F(SinhTest, sinh_test_case_1) {
   ge::op::Sinh sinh_op;
   ge::TensorDesc tensorDesc;
   ge::Shape shape({2, 3, 4});
   tensorDesc.SetDataType(ge::DT_FLOAT16);
   tensorDesc.SetShape(shape);
   tensorDesc.SetOriginShape(shape);
   sinh_op.UpdateInputDesc("x", tensorDesc);
   auto ret = sinh_op.InferShapeAndType();
   EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
   auto output_desc = sinh_op.GetOutputDescByName("y");
   EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
   std::vector<int64_t> expected_output_shape = {2, 3, 4};
   EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
