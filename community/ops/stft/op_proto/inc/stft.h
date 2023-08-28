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


#ifndef GE_OP_STFT_H
#define GE_OP_STFT_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(STFT)
    .INPUT(x, TensorType({DT_FLOAT32}))
    .INPUT(window, TensorType({DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_FLOAT32}))
    .REQUIRED_ATTR(n_fft, Int)
    .ATTR(hop_length, Int, 128)
    .ATTR(win_length, Int, 0)
    .ATTR(center, Bool, false)
    .ATTR(pad_mode, String, "reflect")
    .ATTR(normalized, Bool, false)
    .ATTR(onesided, Bool, true)
    .ATTR(return_complex, Bool, true)
    .OP_END_FACTORY_REG(STFT)
}
#endif //GE_OP_STFT_H