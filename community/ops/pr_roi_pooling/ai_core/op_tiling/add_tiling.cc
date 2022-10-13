/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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

#include "register/op_impl_registry.h"

namespace optiling {
struct AddCompileInfo {
  int32_t block_dim;
  int32_t ub_size;
};

ge::graphStatus TilingPrepare4Add(gert::TilingParseContext* context) {
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4Add(gert::TilingContext* context) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(Add).Tiling(Tiling4Add).TilingParse<AddCompileInfo>(TilingPrepare4Add);
} // namespace optiling