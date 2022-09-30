
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of Add
 */

#ifndef _ADD_KERNELS_H_
#define _ADD_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class AddCpuKernel : public CpuKernel {
public:
    ~AddCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
};
} // namespace aicpu
#endif
