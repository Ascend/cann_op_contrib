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
#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "node_def_builder.h"
#include "aicpu_read_file.h"
#include "utils/kernel_util.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_TANH_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Tanh", "Tanh")                   \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Output({"y", data_types[1], shapes[1], datas[1]})

bool ReadFileFloat(std::string file_name, std::complex<float> output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    for (uint64_t index = 0; index < size; ++index) {
      string s, s1,s2;
      stringstream ss, sss;
      string ::size_type n1, n2,n3;
      bool flag = true;

      getline(in_file, s);
      n1 = s.find("(", 0);
      n2 = s.find("+", 0);
      if (n2 == string::npos) {
        n2 = s.find("-", n1 + 2);
        flag = false;
      }
      n3 = s.find("j", 0);
      s1 = s.substr(n1 + 1, n2 - n1 - 1);
      s2 = s.substr(n2 + 1, n3 - n2 - 1);

     float temp;
      ss << s1;
      ss >> temp;
      output[index].real(temp);
      sss << s2;
      sss >> temp;
     if (!flag)
        temp *= -1;
      output[index].imag(temp);
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

bool ReadFileDouble(std::string file_name, std::complex<double> output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    for (uint64_t index = 0; index < size; ++index) {
      string s, s1,s2;
      stringstream ss, sss;
      string ::size_type n1, n2,n3;
      bool flag = true;

      getline(in_file, s);
      n1 = s.find("(", 0);
      n2 = s.find("+", 0);
      if (n2 == string::npos) {
        n2 = s.find("-", n1 + 2);
        flag = false;
      }
      n3 = s.find("j", 0);
      s1 = s.substr(n1 + 1, n2 - n1 - 1);
      s2 = s.substr(n2 + 1, n3 - n2 - 1);

      float temp;
      ss << s1;
      ss >> temp;
      output[index].real(temp);
      sss << s2;
      sss >> temp;
      if (!flag)
        temp *= -1;
      output[index].imag(temp);
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

bool CompareResult_3(std::complex<float> output[],
                    std::complex<float> expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    if (std::abs(output[i] - expect_output[i]) > 1e-3) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

bool CompareResult_4(std::complex<double> output[],
                    std::complex<double> expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    if (std::abs(output[i] - expect_output[i]) > 1e-3) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

void RunTanhKernelComplexFloat(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  std::complex<float> input1[input1_size];
  bool status = ReadFileFloat(data_path, input1, input1_size);
  EXPECT_EQ(status, true);
  
  uint64_t output_size = CalTotalElements(shapes, 1);
  std::complex<float> output[output_size];

  vector<void *> datas = {(void *)input1,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  std::complex<float> output_exp[output_size];
  status = ReadFileFloat(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult_3(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  
}

void RunTanhKernelComplexDouble(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  std::complex<double> input1[input1_size];
  bool status = ReadFileDouble(data_path, input1, input1_size);
  EXPECT_EQ(status, true);
   
  uint64_t output_size = CalTotalElements(shapes, 1);
  std::complex<double> output[output_size];

  vector<void *> datas = {(void *)input1,
                         
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  std::complex<double> output_exp[output_size];
  status = ReadFileDouble(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult_4(output, output_exp, output_size);
  EXPECT_EQ(compare, true);

}


// read input and output data from files which generate by your python file
template<typename T1, typename T2>
void RunTanhKernel(vector<string> data_files,
                   vector<DataType> data_types,
                   vector<vector<int64_t>> &shapes) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  
  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 *output = new T2[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[1];
  T2 *output_exp = new T2[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}
// only generate input data by SetRandomValue,
// and calculate output by youself function

TEST_F(TEST_TANH_UT, DATA_TYPE_FLOAT1_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{15}, {15}};
  vector<string> files{"tanh/data/tanh_data_input1_1.txt",                     
                       "tanh/data/tanh_data_output1_1.txt"};
  RunTanhKernel<float, float>(files, data_types, shapes);
}

TEST_F(TEST_TANH_UT, DATA_TYPE_FLOAT2_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1250, 1024}, {1250, 1024}};
  vector<string> files{"tanh/data/tanh_data_input1_2.txt",                     
                       "tanh/data/tanh_data_output1_2.txt"};
  RunTanhKernel<float, float>(files, data_types, shapes);
}

TEST_F(TEST_TANH_UT, DATA_TYPE_FLOAT3_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{15, 1025, 30}, {15, 1025, 30}};
  vector<string> files{"tanh/data/tanh_data_input1_3.txt",                     
                       "tanh/data/tanh_data_output1_3.txt"};
  RunTanhKernel<float, float>(files, data_types, shapes);
}

TEST_F(TEST_TANH_UT, DATA_TYPE_DOUBLE1_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1170}, {1170}};
  vector<string> files{"tanh/data/tanh_data_input1_4.txt",                       
                       "tanh/data/tanh_data_output1_4.txt"};
  RunTanhKernel<double, double>(files, data_types, shapes);
}

TEST_F(TEST_TANH_UT, DATA_TYPE_DOUBLE2_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{7, 12}, {7, 12}};
  vector<string> files{"tanh/data/tanh_data_input1_5.txt",
                       
                       "tanh/data/tanh_data_output1_5.txt"};
  RunTanhKernel<double, double>(files, data_types, shapes);
}

TEST_F(TEST_TANH_UT, DATA_TYPE_DOUBLE3_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{7, 1200, 30}, {7, 1200, 30}};
  vector<string> files{"tanh/data/tanh_data_input1_6.txt",
                       
                       "tanh/data/tanh_data_output1_6.txt"};
  RunTanhKernel<double, double>(files, data_types, shapes);
}

TEST_F(TEST_TANH_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{256, 1024}, {256, 1024}};
  vector<string> files{"tanh/data/tanh_data_input1_7.txt",
                       
                       "tanh/data/tanh_data_output1_7.txt"};
  RunTanhKernel<Eigen::half, Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_TANH_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{1300, 10, 4}, {1300, 10, 4}};
  vector<string> files{"tanh/data/tanh_data_input1_8.txt",
                       
                       "tanh/data/tanh_data_output1_8.txt"};
   RunTanhKernelComplexFloat(files, data_types, shapes);
}

TEST_F(TEST_TANH_UT, DATA_TYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{6, 6, 6}, {6, 6, 6}};
  vector<string> files{"tanh/data/tanh_data_input1_9.txt",
                       
                       "tanh/data/tanh_data_output1_9.txt"};
  RunTanhKernelComplexDouble(files, data_types, shapes);
}
// exception instance

TEST_F(TEST_TANH_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  int32_t input1[22] = {(int32_t)1};
  int64_t output[22] = {(int64_t)0};
  vector<void *> datas = {(void *)input1,  (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TANH_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  int32_t output[22] = {(int32_t)0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TANH_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11},  {2, 11}};
  bool input1[22] = {(bool)1};

  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1,  (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
