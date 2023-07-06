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

#ifndef COMMUNITY_COMMON_UTILS_ASCENDC_UT_UTIL_H_
#define COMMUNITY_COMMON_UTILS_ASCENDC_UT_UTIL_H_
#include <fstream>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdio.h>
#include<iostream>

#include "tikicpulib.h"
const std::string ktestcaseFilePath =
    "../community/tests/";

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

bool ReadFile(const std::string& filePath, size_t& fileSize, void* buffer, size_t bufferSize);
bool WriteFile(const std::string& filePath, const void* buffer, size_t size);
bool CompareResult(half output[], half expect_output[], uint64_t num);
bool ReadFile(std::string file_name, half output[], uint64_t size);

template <typename T>
bool CompareResult(T output[], T expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    if (output[i] != expect_output[i]) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}
#endif