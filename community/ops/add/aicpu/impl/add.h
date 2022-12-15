/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
#ifndef _ADD_KERNELS_H_
#define _ADD_KERNELS_H_

#include "cpu_kernel.h"
#include "cpu_types.h"
#include "utils/bcast.h"

namespace aicpu {
class AddCpuKernel : public CpuKernel {
 public:
  AddCpuKernel() = default;
  ~AddCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  /**
   * @brief compute for all types
   * @param ctx cpu kernel context
   * @return status if success
   */
  template <typename T>
  uint32_t AddCompute(const CpuKernelContext &ctx);

  /**
   * @brief Check if input&output addr is aligned
   * @param calcInfo data used to calculate
   * @return true: aligned, false: not aligned
   */
  bool AlignedCheck(const BCalcInfo &calcInfo) const;
  
  template <int32_t RANK, typename T>
  uint32_t AddCalculateWithAlignedCheck(const CpuKernelContext &ctx, BCalcInfo &calcInfo);

  /**
   * @brief Eigen calculate for all types
   * @param calcInfo data used to calculate
   */
  template <int32_t RANK, typename T, int32_t OPTION>
  uint32_t AddCalculate(BCalcInfo &calcInfo);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_ADD_H_