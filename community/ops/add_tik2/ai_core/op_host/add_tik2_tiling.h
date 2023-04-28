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

#ifndef ADD_TIK2_TILING_H
#define ADD_TIK2_TILING_H

#ifdef TIK2_UT

struct AddTik2TilingData {
    uint32_t blockDim;
    uint32_t totalLength;
    uint32_t tileNum;
};

#define GET_TILING_DATA(tilingData, tilingPointer)                                                             \
    AddTik2TilingData *tilingDataPointer = reinterpret_cast<AddTik2TilingData *>((uint8_t *)(tilingPointer));  \
    AddTik2TilingData tilingData(*tilingDataPointer);

#else // TIK2_UT
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(AddTik2TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddTik2, AddTik2TilingData)
}
#endif // TIK2_UT
#endif // ADD_TIK2_TILING_H
