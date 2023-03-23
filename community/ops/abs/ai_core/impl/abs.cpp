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

#include "tikcfw/kernel_operator.h"

namespace tik2 {
class KernelAbs {
public:
    __aicore__ inline KernelAbs() {}
    __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dst)
    {
        srcGlobal.SetGlobalBuffer((__gm__ half*)src);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dst);
        pipe.InitBuffer(inQueueSrc, 1, dataSize * sizeof(half));
        pipe.InitBuffer(outQueueDst, 1, dataSize * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
        DataCopy(srcLocal, srcGlobal, dataSize);
        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
        LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
        Abs(dstLocal, srcLocal, dataSize);
        outQueueDst.EnQue<half>(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
        DataCopy(dstGlobal, dstLocal, dataSize);
        outQueueDst.FreeTensor(dstLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueSrc;
    TQue<QuePosition::VECOUT, 1> outQueueDst;
    GlobalTensor<half> srcGlobal, dstGlobal;
    int32_t dataSize = 512;
};
} // namespace tik2
extern "C" __global__ __aicore__ void abs_tik2(__gm__ uint8_t* x, __gm__ uint8_t* y)
{
    tik2::KernelAbs op;
    op.Init(x, y);
    op.Process();
}
