'''
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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
'''


def fuzz_branch():
    # input
    rois_n = 128
    roi_value = [[0,0,0,10,12] for i in range(64)] + [[3,2,2,8,10] for i in range(64)]

    return {
        "input_desc": {
            "rois": {"value": roi_value, "shape": [rois_n, 5]},
        },
        "output_desc": {
            "y": {"shape": [rois_n,1024,8,8]}
        }          
    }
