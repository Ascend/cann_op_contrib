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
#include "tik2_ut_util.h"

bool ReadFile(const std::string& filePath, size_t& fileSize, void* buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file%s",filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }
    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }
    std::filebuf* buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char*>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

bool WriteFile(const std::string& filePath, const void* buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }
    auto writeSize = write(fd, buffer, size);
    (void)close(fd);
    if (writeSize != static_cast<int64_t>(size)) {
        ERROR_LOG("Write file Failed.");
        return false;
    }
    return true;
}

bool CompareResult(half output[], half expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    half absolute_error = (output[i] - expect_output[i]);
    absolute_error =
        absolute_error >= half(0) ? static_cast<float>(absolute_error) : -static_cast<float>(absolute_error);
    half relative_error(0);
    if (expect_output[i] == half(0)) {
      relative_error = half(2e-3);
    } else {
      relative_error = absolute_error / expect_output[i];
      relative_error =
          relative_error >= half(0) ? static_cast<float>(relative_error) : -static_cast<float>(relative_error);
    }
    if ((absolute_error > half(1e-3)) &&
        (relative_error > half(1e-3))) {
      std::cout << "output[" << i << "] = ";
      std::cout << static_cast<float>(output[i]);
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << static_cast<float>(expect_output[i]) << std::endl;
      result = false;
    }
  }
  return result;
}

bool CompareResult(float output[], float expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    double absolute_error = std::fabs(output[i] - expect_output[i]);
    double relative_error = 0;
    if (expect_output[i] == 0) {
      relative_error = 2e-6;
    } else {
      relative_error = absolute_error / std::fabs(expect_output[i]);
    }
    if ((absolute_error > 1e-6) && (relative_error > 1e-6)) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

bool CompareResult(double output[], double expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    double absolute_error = std::fabs(output[i] - expect_output[i]);
    double relative_error = 0;
    if (expect_output[i] == 0) {
      relative_error = 2e-10;
    } else {
      relative_error = absolute_error / std::fabs(expect_output[i]);
    }
    if ((absolute_error > 1e-12) && (relative_error > 1e-10)) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

bool ReadFile(std::string file_name, half output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." <<std::endl;
      return false;
    }
    float tmp;
    uint64_t index = 0;
    while (in_file >> tmp) {
      if (index >= size) {
        break;
      }
      output[index] = static_cast<half>(tmp);
      index++;
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, "
              << e.what() << std::endl;
    return false;
  }
  return true;
}