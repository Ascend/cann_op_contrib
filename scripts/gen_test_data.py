# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import sys

os.chdir("community/tests")
ut_type = sys.argv[1]

for root, dirs, files in os.walk("./"):
    for file_name in files:
        if "gen_data.py" in file_name:
            file_path = os.path.join(root, file_name)
            cmd = "python3 {}".format(file_path)
            op_name = file_path.split("/")[1]
            if ut_type in file_path:
                print("Gen (", op_name, ") data")
                os.system(cmd)
