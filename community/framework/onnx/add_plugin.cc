/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "external/register/register.h"

namespace domi {
static Status ParseParamsGather(const Message* op_src, ge::Operator& op_dest) {
  return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({"ai.onnx::8::Gather",
  "ai.onnx::9::Gather",
  "ai.onnx::10::Gather",
  "ai.onnx::11::Gather",
  "ai.onnx::12::Gather",
  "ai.onnx::13::Gather",
  "ai.onnx::14::Gather",
  "ai.onnx::15::Gather",
  "ai.onnx::16::Gather"})
  .ParseParamsFn(ParseParamsGather)
  .ImplyType(ImplyType::TVM);
} // namespace domi
