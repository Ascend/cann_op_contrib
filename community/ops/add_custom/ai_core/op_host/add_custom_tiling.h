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

#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H

#ifdef ASCENDC_UT

struct AddCustomTilingData {
    uint32_t blockDim;
    uint32_t totalLength;
    uint32_t tileNum;
};

#define GET_TILING_DATA(tilingData, tilingPointer)                                                                 \
    AddCustomTilingData *tilingDataPointer = reinterpret_cast<AddCustomTilingData *>((uint8_t *)(tilingPointer));  \
    AddCustomTilingData tilingData(*tilingDataPointer);

#else // ASCENDC_UT
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(AddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)
}
#endif // ASCENDC_UT
#endif // ADD_CUSTOM_TILING_H
