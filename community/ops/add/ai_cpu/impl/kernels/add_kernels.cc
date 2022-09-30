
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of Add
 */
#include "add_kernels.h"

namespace  {
const char *ADD = "Add";
}

namespace aicpu  {
uint32_t AddCpuKernel::Compute(CpuKernelContext &ctx)
{
    return 0;
}

REGISTER_CPU_KERNEL(ADD, AddCpuKernel);
} // namespace aicpu
