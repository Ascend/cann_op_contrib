/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

namespace {

template <typename T>
bool CompareRealResult(T output[], T expect_output[], uint64_t num,
                       double epsilon) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    if (abs(output[i] - expect_output[i]) > epsilon) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

}  // namespace

class TEST_CholeskyGrad_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "CholeskyGrad", "CholeskyGrad")   \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Input({"grad", data_types[1], shapes[1], datas[1]})         \
      .Output({"y", data_types[2], shapes[2], datas[2]})

// read input and output data from files which generate by your python file
template <typename T>
void RunCholeskyGradKernel(vector<string> data_files,
                               vector<DataType> data_types,
                               vector<vector<int64_t>> &shapes) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T *a = new T[input1_size];
  bool status = ReadFile(data_path, a, input1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T *L = new T[input2_size];
  status = ReadFile(data_path, L, input2_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 2);
  T *x = new T[output_size];
  vector<void *> datas = {(void *)a, (void *)L, (void *)x};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[2];
  T *x_exp = new T[output_size];
  status = ReadFile(data_path, x_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareRealResult(x, x_exp, output_size, 1e-4);
  EXPECT_EQ(compare, true);
  delete[] a;
  delete[] L;
  delete[] x;
  delete[] x_exp;
}

template <typename T>
void RunCholeskyGradKernel2(vector<DataType> data_types,
                                vector<vector<int64_t>> &shapes) {
  // gen data use SetRandomValue for input1
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T *a = new T[input1_size];
  SetRandomValue<T>(a, input1_size);

  uint64_t input2_size = CalTotalElements(shapes, 1);
  T *L = new T[input2_size];
  SetRandomValue<T>(L, input2_size);

  uint64_t output_size = CalTotalElements(shapes, 2);
  T *x = new T[output_size];
  vector<void *> datas = {(void *)a, (void *)L, (void *)x};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // calculate output_exp
  T *x_exp = new T[output_size];
  *(x_exp) = *(L)/ (2.* *(a));

  bool compare = CompareRealResult(x, x_exp, output_size, 1e-4);
  EXPECT_EQ(compare, true);
  delete[] a;
  delete[] L;
  delete[] x;
  delete[] x_exp;
}

TEST_F(TEST_CholeskyGrad_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{16, 16}, {16, 16}, {16, 16}};
  vector<string> files{
      "choleskyGrad/data/choleskyGrad_data_input1_double.txt",
      "choleskyGrad/data/choleskyGrad_data_input2_double.txt",
      "choleskyGrad/data/choleskyGrad_data_output1_double.txt"};
  RunCholeskyGradKernel<double>(files, data_types, shapes);
}

TEST_F(TEST_CholeskyGrad_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{16, 16}, {16, 16}, {16, 16}};
  vector<string> files{
      "choleskyGrad/data/choleskyGrad_data_input1_float.txt",
      "choleskyGrad/data/choleskyGrad_data_input2_float.txt",
      "choleskyGrad/data/choleskyGrad_data_output1_float.txt"};
  RunCholeskyGradKernel<float>(files, data_types, shapes);
}

TEST_F(TEST_CholeskyGrad_UT, DATA_TYPE_FLOAT_BIG_DATA_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{512, 512}, {512, 512}, {512, 512}};
  vector<string> files{
      "choleskyGrad/data/choleskyGrad_data_input1_float_1.txt",
      "choleskyGrad/data/choleskyGrad_data_input2_float_1.txt",
      "choleskyGrad/data/choleskyGrad_data_output1_float_1.txt"};
  RunCholeskyGradKernel<float>(files, data_types, shapes);
}

TEST_F(TEST_CholeskyGrad_UT, DATA_TYPE_FLOAT_HIGH_DIM_1_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 4, 4}, {2, 4, 4}};
  vector<string> files{
      "choleskyGrad/data/choleskyGrad_data_input1_3d_1.txt",
      "choleskyGrad/data/choleskyGrad_data_input2_3d_1.txt",
      "choleskyGrad/data/choleskyGrad_data_output1_3d_1.txt"};
  RunCholeskyGradKernel<float>(files, data_types, shapes);
}

TEST_F(TEST_CholeskyGrad_UT, DATA_TYPE_FLOAT_HIGH_DIM_2_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{32, 8, 8}, {32, 8, 8}, {32, 8, 8}};
  vector<string> files{
      "choleskyGrad/data/choleskyGrad_data_input1_3d_2.txt",
      "choleskyGrad/data/choleskyGrad_data_input2_3d_2.txt",
      "choleskyGrad/data/choleskyGrad_data_output1_3d_2.txt"};
  RunCholeskyGradKernel<float>(files, data_types, shapes);
}

TEST_F(TEST_CholeskyGrad_UT, DATA_TYPE_FLOAT_HIGH_DIM_3_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{128, 16, 16}, {128, 16, 16}, {128, 16, 16}};
  vector<string> files{
      "choleskyGrad/data/choleskyGrad_data_input1_3d_3.txt",
      "choleskyGrad/data/choleskyGrad_data_input2_3d_3.txt",
      "choleskyGrad/data/choleskyGrad_data_output1_3d_3.txt"};
  RunCholeskyGradKernel<float>(files, data_types, shapes);
}

TEST_F(TEST_CholeskyGrad_UT, DATA_TYPE_DT_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 1}, {1, 1}, {1, 1}};
  RunCholeskyGradKernel2<float>(data_types, shapes);
}

TEST_F(TEST_CholeskyGrad_UT, DATA_TYPE_DT_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1, 1}, {1, 1}, {1, 1}};
  RunCholeskyGradKernel2<double>(data_types, shapes);
}

TEST_F(TEST_CholeskyGrad_UT, DATA_TYPE_FLOAT_UNSPD_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2, 2}};
  vector<string> files{
      "choleskyGrad/data/choleskyGrad_data_input1_float_unspd.txt",
      "choleskyGrad/data/choleskyGrad_data_input2_float_unspd.txt",
      "choleskyGrad/data/choleskyGrad_data_output1_float_unspd.txt"};
  RunCholeskyGradKernel<float>(files, data_types, shapes);
}

// exception instance
TEST_F(TEST_CholeskyGrad_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2, 1}};
  double a[4] = {1.};
  double L[4] = {1.};
  double output[2] = {1.};
  vector<void *> datas = {(void *)a, (void *)L, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CholeskyGrad_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_FLOAT, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2, 2}};
  double a[4] = {1.};
  double L[4] = {1.};
  double output[4] = {1.};
  vector<void *> datas = {(void *)a, (void *)L, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CholeskyGrad_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2, 2}};
  double a[4] = {1.};
  double output[4] = {1.};
  vector<void *> datas = {(void *)a, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CholeskyGrad_UT, INPUT_ZERO_DIM_NOT_TWO) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 2, 1}, {2, 2}, {2, 2}};
  double a[4] = {1.};
  double L[4] = {1.};
  double output[4] = {1.};
  vector<void *> datas = {(void *)a, (void *)L, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CholeskyGrad_UT, INPUT_ONE_DIM_NOT_TWO) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2, 1}, {2, 2}};
  double a[4] = {1.};
  double L[4] = {1.};
  double output[4] = {1.};
  vector<void *> datas = {(void *)a, (void *)L, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CholeskyGrad_UT, OUTPUT_ZERO_DIM_NOT_TWO) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2, 2, 1}};
  double a[4] = {1.};
  double L[4] = {1.};
  double output[4] = {1.};
  vector<void *> datas = {(void *)a, (void *)L, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CholeskyGrad_UT, INPUT_ZERO_NOT_SQUARE) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 2}, {2, 2}};
  double a[6] = {1.};
  double L[4] = {1.};
  double output[4] = {1.};
  vector<void *> datas = {(void *)a, (void *)L, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CholeskyGrad_UT, INPUT_INPUT_MISMATCH) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 2}, {3, 3}, {2, 2}};
  double a[4] = {1.};
  double L[9] = {1.};
  double output[4] = {1.};
  vector<void *> datas = {(void *)a, (void *)L, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CholeskyGrad_UT, INPUT_OUTPUT_MISMATCH) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {3, 3}};
  double a[4] = {1.};
  double L[4] = {1.};
  double output[9] = {1.};
  vector<void *> datas = {(void *)a, (void *)L, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CholeskyGrad_UT, DATA_TYPE_NOT_SUPPORT) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2, 2}};
  double a[4] = {1.};
  double L[4] = {1.};
  double output[4] = {1.};
  vector<void *> datas = {(void *)a, (void *)L, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}