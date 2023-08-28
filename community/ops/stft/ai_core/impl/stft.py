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

from te import tik
from tbe.common.platform import set_current_compile_soc_info, get_soc_spec
from numpy import pi, cos, sin, log2, sqrt
from functools import reduce

class STFT:
    def __init__(self, x, window, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided, y, kernel_name):
        """
        The STFT computes the Fourier transform of short overlapping windows of the input.
        @param input Indicates input tensor information
        @param output Indicates output tensor information
        @param n_fft Indicates size of Fourier transform
        @param hop_length Indicates the distance between neighboring sliding window frames
        @param win_length Indicates the size of window frame and STFT filter
        @param window Indicates the optional window function
        @param center Indicates if input will be padded
        @param pad_mode If centered -  indicates which padding mode to use
        @param kernel_name Indicates the kernel name of the function in the generated binary code

        """
        self.input = x
        self.window = window
        self.output = y
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length != 0 else n_fft
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided
        self.kernel_name = kernel_name

        self.input_shape = x.get("shape")
        self.input_dtype = x.get("dtype")
        self.output_shape = y.get("shape")
        self.output_dtype = y.get("dtype")
        self.window_shape = window.get("shape")
        self.window_dtype = window.get("dtype")

        self.n = self.input_shape[-2]
        self.batches = self.input_shape[-3]        

        self._input_check()
        self._window_check()
        self._param_check()

        self.frame_count = (self.n - self.n_fft) // self.hop_length + 1
        if center:
            self.frame_count = self.n // self.hop_length + 1
        self.overall_frame_count = self.frame_count * self.batches

        self.norm = 1 / sqrt(self.n_fft)
        self._output_check()

        if center:
            self.pad_mode = self.pad_mode.lower()
        self.factors = self._factorize()

        self.pure_power_of_two = True

        for factor in self.factors:
            if factor not in (2,4,8,16):
                self.pure_power_of_two = False
                break
        

        self.n_coeff, self.coeff_real_init_value, self.coeff_imag_init_value = self._dft_coeff()

        self.ascend910b = (get_soc_spec("SHORT_SOC_VERSION") == "Ascend910B")
        if self.ascend910b:
            set_current_compile_soc_info("Ascend910B1", "VectorCore")
        else:
            set_current_compile_soc_info("Ascend310P1")

        self.tik_instance = tik.Tik(disable_debug=False)
        
        self.cores_num_910B = get_soc_spec("CORE_NUM")
        self.frames_per_core = self.overall_frame_count // self.cores_num_910B
        self.left_over_frames = self.overall_frame_count % self.cores_num_910B

        self.input_data = self.tik_instance.Tensor(dtype="float32", shape=self.input_shape, name="input_data", scope=tik.scope_gm)
        self.window_data = self.tik_instance.Tensor(dtype="float32", shape=self.window_shape, name="window_data", scope=tik.scope_gm)

        self.sync_workspace = self.tik_instance.Tensor(dtype="int64", shape=(min(self.cores_num_910B, self.overall_frame_count) * 4,), name="sync_workspace", 
                                                       scope=tik.scope_gm, is_workspace =True, is_atomic_add=True)
        self.pre_transpose_data = self.tik_instance.Tensor(dtype="float32", shape=(self.batches, self.frame_count, 2, self.n_fft), name="pre_transpose_data", 
                                                           scope=tik.scope_gm, is_workspace=True)
        self.output_data = self.tik_instance.Tensor(dtype="float32", shape=(self.output_shape), name="output_data", scope=tik.scope_gm)

        if self.ascend910b:
            self.output_tmp_data = self.tik_instance.Tensor(dtype="float32", shape=(self.batches, self.n_fft, self.frame_count,2,), name="output_tmp_data",
                                                            scope=tik.scope_gm, is_workspace=self.onesided)
        else:
            self.output_data_310p = self.tik_instance.Tensor(dtype="float32", shape=(self.output_shape[-4] + 1, self.output_shape[-3], self.output_shape[-2], self.output_shape[-1]), 
                                                              name="output_data_310p", scope=tik.scope_gm)       
            self.output_tmp_data = self.tik_instance.Tensor(dtype="float32", shape=(self.batches + 1, self.n_fft, self.frame_count, 2), name="output_tmp_data",
                                                            scope=tik.scope_gm, is_workspace=self.onesided)
            
        self.coeff_data_real = self.tik_instance.Tensor(dtype="float32", shape=(self.n_coeff,), name="coeff_data_real", scope=tik.scope_gm, 
                                                        init_value=self.coeff_real_init_value)
        self.coeff_data_imag = self.tik_instance.Tensor(dtype="float32", shape=(self.n_coeff,), name="coeff_data_imag", scope=tik.scope_gm, 
                                                        init_value=self.coeff_imag_init_value)
        
        if self.n_fft == 400:
            self.custom_masks_400 = list()
            self._count_custom_masks_400()
        
        self.pad_data = self.tik_instance.Tensor(dtype="float32", shape=(1,), name="dummy_data", scope=tik.scope_gm, is_workspace=True)
        
        if self.center:
            self.pad_data_init_value_overflow = self.batches * 2 * 2 * self.n_fft/2 > 256 * 1024

            if not self.pad_data_init_value_overflow  and self.pad_mode == "constant":
                self.pad_data = self.tik_instance.Tensor(dtype="float32", shape=(self.batches, 2, 2, self.n_fft/2,), name="pad_data", scope=tik.scope_gm, init_value = 0)
            else:
                self.pad_data = self.tik_instance.Tensor(dtype="float32", shape=(self.batches, 2, 2, self.n_fft/2,), name="pad_data", scope=tik.scope_gm, is_workspace=True)
                # batches, 2 - left/right, 2 - complex, n_fft/2
        

    def _is_power_of_two(self, n):
        return n and not (n & (n - 1))

    def _input_check(self):
        if len(self.input_shape) != 3 or self.input_shape[-1] != 2:
            raise ValueError("Input shape doesn't match")
        if self.input_dtype != "float32":
            raise ValueError("Input data type doesn't match")

    def _window_check(self):
        if len(self.window_shape) != 1:
            raise ValueError("Window shape doesn't match")
        if self.window_dtype != "float32":
            raise ValueError("Window data type doesn't match")
        if self.window_shape[-1] != self.win_length:
            raise ValueError("Window length parameter and window shape don't match")
        
    def _param_check(self):
        _is_int = lambda a : type(a) is int

        if not (_is_int(self.n_fft) and _is_int(self.win_length) and _is_int(self.hop_length)):
            raise ValueError("Some of the parameters are not int types")
        if (not self._is_power_of_two(self.n_fft) or (self._is_power_of_two(self.n_fft) and (self.n_fft < 16 or self.n_fft > 8192))) and self.n_fft != 400:
            raise ValueError("n_fft should be power of two in [16:8192] or 400")
        if self.win_length % 16 != 0 or self.win_length > self.n_fft:
            raise ValueError("win_length should be a multiple of 16 and less or equal than n_fft")
        if self.win_length > self.input_shape[-2] or self.hop_length > self.input_shape[-2] or self.n_fft > self.input_shape[-2]:
            raise ValueError("Some of the parameters are bigger than input")
        if (not self._is_power_of_two(self.hop_length) or (self._is_power_of_two(self.hop_length) and (self.hop_length < 8 or self.hop_length > 8192))) and self.hop_length != 160:
            raise ValueError("hop_length should be power of two in [8:8192] or 160")
        if self.win_length <= 0 or self.hop_length <= 0 or self.n_fft <= 0:
            raise ValueError("Some of the parameters are not positive")

        if self.center:
            modes = ("constant", "reflect", "replicate", "circular")

            if type(self.pad_mode) is not str:
                raise ValueError("Pad mode needs to be a string")
            if not self.pad_mode.lower() in modes:
                raise ValueError("Pad mode only supports the following modes: \"constant\", \"reflect\", \"replicate\" and \"circular\"")
    
    def _output_check(self):
        if len(self.output_shape) != 4:
            raise ValueError("Output shape doesn't match")
        if self.output_dtype != "float32":
            raise ValueError("Output data type doesn't match")
        if (self.onesided):
            if self.output_shape[-4] != self.batches or self.output_shape[-3] != (self.n_fft // 2 + 1) or self.output_shape[-2] != self.frame_count or self.output_shape[-1] != 2:
                raise ValueError("Output shape doesn't match")
        else:
            if self.output_shape[-4] != self.batches or self.output_shape[-3] != self.n_fft or self.output_shape[-2] != self.frame_count or self.output_shape[-1] != 2:
                raise ValueError("Output shape doesn't match")
    
    def _count_custom_masks_400(self):
        mask_orig = [0 if (i % 50 or i > 400) else 1 for i in range(416)] #416 is divided by 32
        mask_zeros = [0] * 416

        for i in range(2): #even/odd
            first_list = list()
            for j in range(0, 9, 2): #line start
                second_list = list()
                for k in range(0, 41, 10): #inner block
                    shift_value = i + j + k
                    cur_custom_mask_400 = mask_orig
                    if shift_value != 0:
                        cur_custom_mask_400 = mask_zeros[-shift_value:]+mask_orig[:-shift_value]

                    arr_to_fill = []

                    for l in range(0, 416, 32):
                        arr_tmp = cur_custom_mask_400[l:l+32]
                        if 1 in arr_tmp:
                            arr_to_fill.append(1 << (arr_tmp.index(1)))
                        else:
                            arr_to_fill.append(0)

                    tensor_to_pass = self.tik_instance.Tensor(dtype="uint32", shape=(416 // 32,), name=f"custom_mask_400_{i}_{j}_{k}", scope=tik.scope_gm, init_value=arr_to_fill)
                    second_list.append(tensor_to_pass)
                
                first_list.append(second_list)
            self.custom_masks_400.append(first_list)

    def _count_pads_parallel(self, additional_batch, cores_with_extra_batches, cores_with_normal_batches):
        with self.tik_instance.for_range(0, self.batch_per_core + additional_batch) as cur_i:
            batch_pad_i = (cores_with_extra_batches * (self.batch_per_core + 1)) + (cores_with_normal_batches * self.batch_per_core) + cur_i
            batch_pad_src_offset = batch_pad_i * self.n * 2

            if self.is_additional_pad_space_needed:
                tmp_ub = self.tik_instance.Tensor(dtype="float32", shape = (2, 2, self.n_fft // 2), name="tmp_pad_ub", scope=tik.scope_ubuf)

                self.tik_instance.data_move(tmp_ub, self.input_data[batch_pad_src_offset + 2 * self.is_reflect], 0, 1, 4 * self.n_fft // 32, 0, 0)
                self.tik_instance.data_move(tmp_ub[self.n_fft], self.input_data[batch_pad_src_offset + 2 * self.n - self.n_fft - 2 * self.is_reflect], 0, 1, 4 * self.n_fft // 32, 0, 0)

                left_input_ub = self.tik_instance.Tensor(dtype="float32", shape = (2, self.n_fft // 2), name="left_input_ub", scope=tik.scope_ubuf)
                right_input_ub = self.tik_instance.Tensor(dtype="float32", shape = (2, self.n_fft // 2), name="right_input_ub", scope=tik.scope_ubuf)

                self.tik_instance.vreduce(self.n_fft, left_input_ub, tmp_ub, 1, 1, 1, 8, 0, 0, None, "counter")
                self.tik_instance.vreduce(self.n_fft, left_input_ub[self.n_fft // 2], tmp_ub, 2, 1, 1, 8, 0, 0, None, "counter")
                
                self.tik_instance.vreduce(self.n_fft, right_input_ub, tmp_ub[self.n_fft], 1, 1, 1, 8, 0, 0, None, "counter")
                self.tik_instance.vreduce(self.n_fft, right_input_ub[self.n_fft // 2], tmp_ub[self.n_fft], 2, 1, 1, 8, 0, 0, None, "counter")

                if self.is_reflect:
                    self._count_pad_reflect_case(True, right_input_ub, batch_pad_i)
                    self._count_pad_reflect_case(False, left_input_ub, batch_pad_i)
                else:
                    self._count_pad_circular_case(True, left_input_ub, batch_pad_i)
                    self._count_pad_circular_case(False, right_input_ub, batch_pad_i)

            elif self.is_replicate:
                self._count_pad_replicate_case(batch_pad_i)
            
            elif self.pad_data_init_value_overflow:
                self._count_pad_constant_case(batch_pad_i)
                

    def _count_pad_circular_case(self, is_right, passed_ub, batch_pad_i):
        batch_pad_dst_offset = batch_pad_i * 2 * self.n_fft
        right_dst_offset = is_right * self.n_fft

        self.tik_instance.data_move(self.pad_data[batch_pad_dst_offset + right_dst_offset], passed_ub, 0, 1, (4 * self.n_fft) // 32, 0, 0)
    
    def _count_pad_reflect_case(self, is_right, passed_ub, batch_pad_i):
        batch_pad_dst_offset = batch_pad_i * 2 * self.n_fft
        right_dst_offset = is_right * (self.n_fft)
        
        tmp_data = self.tik_instance.Tensor(dtype="float32", shape = (2, self.n_fft // 2), name="tmp_data_ub", scope=tik.scope_ubuf)

        for is_imag in range(2):
            imag_offset = is_imag * self.n_fft // 2

            for cur_i in range(self.n_fft // 2):
                tmp_data[(imag_offset + self.n_fft // 2 - 1) - cur_i].set_as(passed_ub[imag_offset + cur_i])
        
        self.tik_instance.data_move(self.pad_data[batch_pad_dst_offset + right_dst_offset], tmp_data, 0, 1, self.n_fft * 4 // 32, 0, 0)
    
    def _count_pad_replicate_case(self, batch_pad_i):
        batch_pad_dst_offset = batch_pad_i * 2 * self.n_fft

        left_real = self.tik_instance.Scalar(dtype="float32", init_value=self.input_data[batch_pad_i * 2 * self.n + 0])
        left_imag = self.tik_instance.Scalar(dtype="float32", init_value=self.input_data[batch_pad_i * 2 * self.n + 1])
        right_real = self.tik_instance.Scalar(dtype="float32", init_value=self.input_data[batch_pad_i * 2 * self.n + 2 * self.n - 2])
        right_imag = self.tik_instance.Scalar(dtype="float32", init_value=self.input_data[batch_pad_i * 2 * self.n + 2 * self.n - 1])

        tmp_ub_pad = self.tik_instance.Tensor(dtype="float32", shape=(2, 2, self.n_fft // 2,), name="tmp_ub_pad", scope=tik.scope_ubuf)
        mask = min(64, self.n_fft // 2)
        vec_dup_repeat_times = max(1, self.n_fft // 2 // 64)
        mask_leftover = (self.n_fft // 2) % 64 if vec_dup_repeat_times > 1 else 0
        elems_multiplyed = mask * vec_dup_repeat_times

        self.tik_instance.vec_dup(mask, tmp_ub_pad[0], left_real, vec_dup_repeat_times, 8)
        self.tik_instance.vec_dup(mask, tmp_ub_pad[self.n_fft // 2], left_imag, vec_dup_repeat_times, 8)
        self.tik_instance.vec_dup(mask, tmp_ub_pad[self.n_fft], right_real, vec_dup_repeat_times, 8)
        self.tik_instance.vec_dup(mask, tmp_ub_pad[self.n_fft // 2 + self.n_fft], right_imag, vec_dup_repeat_times, 8)

        if mask_leftover > 0:
            self.tik_instance.vec_dup(mask_leftover, tmp_ub_pad[elems_multiplyed], left_real, 1, 8)
            self.tik_instance.vec_dup(mask_leftover, tmp_ub_pad[(self.n_fft // 2) + elems_multiplyed], left_imag, 1, 8)
            self.tik_instance.vec_dup(mask_leftover, tmp_ub_pad[self.n_fft + elems_multiplyed], right_real, 1, 8)
            self.tik_instance.vec_dup(mask_leftover, tmp_ub_pad[(self.n_fft // 2) + self.n_fft + elems_multiplyed], right_imag, 1, 8)

        self.tik_instance.data_move(self.pad_data[batch_pad_dst_offset], tmp_ub_pad, 0, 1, (4 * 2 *self.n_fft) // 32, 0, 0)

    def _count_pad_constant_case(self, batch_pad_i):
        batch_pad_dst_offset = batch_pad_i * 2 * self.n_fft

        tmp_ub_pad = self.tik_instance.Tensor(dtype="float32", shape=(2, 2, self.n_fft // 2,), name="tmp_ub_pad", scope=tik.scope_ubuf)

        mask = min(64, self.n_fft // 2)
        vec_dup_repeat_times = max(1, self.n_fft // 2 // 64)
        mask_leftover = (self.n_fft // 2) % 64 if vec_dup_repeat_times > 1 else 0
        elems_multiplyed = mask * vec_dup_repeat_times

        for i in range(4):
            self.tik_instance.vec_dup(mask, tmp_ub_pad[i * self.n_fft // 2], 0, vec_dup_repeat_times, 8)
            if mask_leftover > 0:
                self.tik_instance.vec_dup(mask_leftover, tmp_ub_pad[i * self.n_fft // 2 + elems_multiplyed], 0, 1, 8)

        self.tik_instance.data_move(self.pad_data[batch_pad_dst_offset], tmp_ub_pad, 0, 1, (4 * 2 *self.n_fft) // 32, 0, 0)

    def _fft_frame(self, frame_i, data_real_ub, data_imag_ub, window_data_ub, coeff_data_real_ub, coeff_data_imag_ub, tmp_ub, custom_masks_ub = None):
        batch_i = frame_i // self.frame_count

        if self.center:
            batch_offset = batch_i * self.n
            cur_frame_in_batch = frame_i % self.frame_count

            center_point = batch_offset + cur_frame_in_batch * self.hop_length

            left_point = center_point - self.n_fft // 2
            right_point = center_point + self.n_fft // 2
            
            with self.tik_instance.if_scope(tik.any(left_point < batch_i * self.n, right_point > (batch_i * self.n + self.n))):
                self._move_pad_data(data_real_ub, data_imag_ub, left_point, right_point, batch_i)
            
            with self.tik_instance.else_scope():
                if self.n_fft >= 4096:
                    tmp_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft * 2,), name="tmp_ub", scope=tik.scope_ubuf)

                self.tik_instance.data_move(tmp_ub, self.input_data[left_point * 2: right_point * 2], 0, 1, self.n_fft * 2 * 4 // 32, 0, 0)
                
                mask = self.n_fft * 2

                self.tik_instance.vreduce(mask, data_real_ub, tmp_ub, 1, 1, 1, 8, 0, 0, None, "counter")
                self.tik_instance.vreduce(mask, data_imag_ub, tmp_ub, 2, 1, 1, 8, 0, 0, None, "counter")
        else:
            with self.tik_instance.new_stmt_scope():
                if self.n_fft >= 4096:
                    tmp_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft * 2,), name="tmp_ub", scope=tik.scope_ubuf)
                
                batched_frame_i = frame_i - batch_i * self.frame_count

                center_point = batch_i * self.n + batched_frame_i * self.hop_length

                self.tik_instance.data_move(tmp_ub, self.input_data[2 * center_point], 0, 1, self.n_fft * 2 * 4 // 32, 0, 0)

                mask = self.n_fft * 2
                self.tik_instance.vreduce(mask, data_real_ub, tmp_ub, 1, 1, 1, 8, 0, 0, None, "counter")
                self.tik_instance.vreduce(mask, data_imag_ub, tmp_ub, 2, 1, 1, 8, 0, 0, None, "counter")

        self._window_mul(data_real_ub, data_imag_ub, window_data_ub)
        self._fft1d(data_real_ub, data_imag_ub, coeff_data_real_ub, coeff_data_imag_ub, custom_masks_ub)
        
        if self.normalized:
            self.tik_instance.vec_muls(self.mask, data_real_ub, data_real_ub, self.norm, self.vec_repeat_times, 8, 8)
            self.tik_instance.vec_muls(self.mask, data_imag_ub, data_imag_ub, self.norm, self.vec_repeat_times, 8, 8)
            if self.mask_leftover:
                    self.tik_instance.vec_muls(self.mask_leftover, data_real_ub[self.elems_multiplyed:], data_real_ub[self.elems_multiplyed:], self.norm, 1, 8, 8)
                    self.tik_instance.vec_muls(self.mask_leftover, data_imag_ub[self.elems_multiplyed:], data_imag_ub[self.elems_multiplyed:], self.norm, 1, 8, 8)

        dst_index_real = frame_i * self.n_fft * 2
        dst_index_imag = dst_index_real + self.n_fft
        
        self.tik_instance.data_move(self.pre_transpose_data[dst_index_real:], data_real_ub, 0, 1, self.n_fft * 4 // 32, 0, 0)
        self.tik_instance.data_move(self.pre_transpose_data[dst_index_imag:], data_imag_ub, 0, 1, self.n_fft * 4 // 32, 0, 0)

    def _data_move_pad_310p(self, dst, src, elements, is_gm = False):
        with self.tik_instance.if_scope(elements >= 8):
            self.tik_instance.data_move(dst, src, 0, 1, elements * 4 // 32, 0, 0)
        
        set_as_offset = (elements * 4 // 32) * 4
        left_over_elements = elements % 8

        if not is_gm:
            with self.tik_instance.for_range(0, left_over_elements) as i:
                cur_offset = set_as_offset + i
                dst[cur_offset].set_as(src[cur_offset])
        else:
            if left_over_elements:
                tmp_ub = self.tik_instance.Tensor(dtype="float32", shape=(8,), name="tmp_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(tmp_ub, dst[set_as_offset], 0, 1, 1, 0, 0)
                for i in range(left_over_elements):
                    cur_offset = set_as_offset + i
                    tmp_ub[i].set_as(src[cur_offset])
                self.tik_instance.data_move(dst[set_as_offset], tmp_ub, 0, 1, 1, 0, 0)
            
    def _move_pad_data(self, data_real_ub, data_imag_ub, left_point, right_point, batch_i):
        with self.tik_instance.if_scope(left_point < batch_i * self.n):
            pad_elements = batch_i * self.n - left_point
            main_elements = self.n_fft - pad_elements
            src_real_index = self.n_fft // 2 - pad_elements

            with self.tik_instance.if_scope(pad_elements % 8 == 0):
                self.tik_instance.data_move(data_real_ub, self.pad_data[batch_i * self.n_fft * 2 + src_real_index], 0, 1, 
                                            pad_elements * 4 // 32, 0, 0)
                self.tik_instance.data_move(data_imag_ub, self.pad_data[batch_i * self.n_fft * 2 + main_elements], 0, 1, 
                                            pad_elements * 4 // 32, 0, 0)
            with self.tik_instance.else_scope():
                if self.ascend910b:
                    self.tik_instance.data_move_pad(data_real_ub, self.pad_data[batch_i * self.n_fft * 2 + src_real_index], 1, pad_elements * 4, 0, 0, 0, 0)
                    self.tik_instance.data_move_pad(data_imag_ub, self.pad_data[batch_i * self.n_fft * 2 + main_elements], 1, pad_elements * 4, 0, 0, 0, 0)
                else:
                    self._data_move_pad_310p(data_real_ub, self.pad_data[batch_i * self.n_fft * 2 + src_real_index], pad_elements)
                    self._data_move_pad_310p(data_imag_ub, self.pad_data[batch_i * self.n_fft * 2 + main_elements], pad_elements)


            tmp_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft * 2,), name="tmp_ub", scope=tik.scope_ubuf)
            with self.tik_instance.if_scope((main_elements * 2) % 8 == 0):
                self.tik_instance.data_move(tmp_ub, self.input_data[batch_i * self.n * 2], 0, 1, main_elements * 2 * 4 // 32, 0, 0)
            with self.tik_instance.else_scope():
                if self.ascend910b:
                    self.tik_instance.data_move_pad(tmp_ub, self.input_data[batch_i * self.n * 2], 1, main_elements * 2 * 4, 0, 0, 0, 0)
                else:
                    self._data_move_pad_310p(tmp_ub, self.input_data[batch_i * self.n * 2], main_elements * 2)

            mask = main_elements * 2
            self.tik_instance.vreduce(mask, data_real_ub[pad_elements], tmp_ub, 1, 1, 1, 8, 0, 0, None, "counter")
            self.tik_instance.vreduce(mask, data_imag_ub[pad_elements], tmp_ub, 2, 1, 1, 8, 0, 0, None, "counter")
        
        with self.tik_instance.else_scope():
            pad_elements = right_point - self.n - batch_i * self.n
            main_elements = self.n_fft - pad_elements

            tmp_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft * 2,), name="tmp_ub", scope=tik.scope_ubuf)
            with self.tik_instance.if_scope((main_elements * 2) % 8 == 0):
                self.tik_instance.data_move(tmp_ub, self.input_data[left_point * 2], 0, 1, main_elements * 2 * 4 // 32, 0, 0)
            with self.tik_instance.else_scope():
                if self.ascend910b:
                    self.tik_instance.data_move_pad(tmp_ub, self.input_data[left_point * 2], 1, main_elements * 2 * 4, 0, 0, 0, 0)
                else:
                    self._data_move_pad_310p(tmp_ub, self.input_data[left_point * 2], main_elements * 2)

            mask = main_elements * 2
            self.tik_instance.vreduce(mask, data_real_ub, tmp_ub, 1, 1, 1, 8, 0, 0, None, "counter")
            self.tik_instance.vreduce(mask, data_imag_ub, tmp_ub, 2, 1, 1, 8, 0, 0, None, "counter")

            with self.tik_instance.if_scope(pad_elements % 8 == 0):
                self.tik_instance.data_move(data_real_ub[main_elements], self.pad_data[batch_i * self.n_fft * 2 + self.n_fft], 0, 1, pad_elements * 4 // 32, 0, 0)
                self.tik_instance.data_move(data_imag_ub[main_elements], self.pad_data[batch_i * self.n_fft * 2 + self.n_fft + self.n_fft // 2], 0, 1, pad_elements * 4 // 32, 0, 0)
            with self.tik_instance.else_scope():
                pad_data_offset = batch_i * self.n_fft * 2 + self.n_fft
                with self.tik_instance.for_range(main_elements, main_elements + pad_elements) as i:
                    data_real_ub[i].set_as(self.pad_data[pad_data_offset + (i - main_elements)])
                    data_imag_ub[i].set_as(self.pad_data[pad_data_offset + self.n_fft // 2 + (i - main_elements)])

    def _factorize(self):
        _n = self.n_fft
        factors = []
        if _n % 8 == 0:
            factors.append(8)
            _n = _n // 8
        while _n % 5 == 0:
            factors.append(5)
            _n = _n // 5
        #while _n % 4 == 0:
        #    factors.append(4)
        #    _n = _n // 4
        while _n % 2 == 0:
            factors.append(2)
            _n = _n // 2
        return factors


    def _dft_coeff(self):
        param = []
        for i in range(len(self.factors)):
            factor = self.factors[i]
            if i == 0:
                if factor == 8:
                    param += [-2 * pi * i * j / 8 for i in range(0, 8) for j in range(0, 8)]
                else:
                    raise ValueError()
            else:
                n_merge = reduce(lambda x, y: x * y, self.factors[:i + 1])
                param += [-2 * pi * i * j / n_merge for i in range(1, factor) for j in range(0, n_merge // factor) ]

        real = cos(param).tolist()
        imag = sin(param).tolist()
        return len(param), real, imag
    
    def _permute_fft_implementation_400(self, tensor, custom_masks_ub):
        tmp_tensor = self.tik_instance.Tensor(dtype="float32", shape=(416,), name="tmp_tensor", scope=tik.scope_ubuf)
        self.tik_instance.data_move(tmp_tensor, tensor, 0, 1, self.n_fft * 4 // 32, 0, 0)

        tmp_ub = self.tik_instance.Tensor(dtype="float32", shape=(8,), name="tmp_ub", scope=tik.scope_ubuf)

        for i in range(50):
            self.tik_instance.vreduce(self.n_fft, tmp_ub, tmp_tensor, custom_masks_ub[i * 16 : i * 16 + 16], 1, 1, 8, 1, 0, None, "counter")

            self.tik_instance.data_move(tensor[i * 8:], tmp_ub, 0, 1, 1, 0, 0)

    def _permute_fft_implementation_p2(self, tensor):
        with self.tik_instance.new_stmt_scope():
            tmp_ub_1 = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft // 2,), name="tmp_ub_1", scope=tik.scope_ubuf)
            tmp_ub_2 = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft // 2,), name="tmp_ub_2", scope=tik.scope_ubuf)

            with self.tik_instance.for_range(0, int(log2(self.n_fft)) - 3) as i:
                nburst = self.tik_instance.Scalar(dtype="uint16", init_value=1)
                nburst.set_as(nburst << i)
                burst_elem = (self.n_fft // 2) // nburst
                burst = burst_elem * 4 // 32
                src_stride = 0
                dst_stride = burst_elem * 4 // 32

                self.tik_instance.vreduce(self.n_fft, tmp_ub_1, tensor, 1, 1, 1, 8, 0, 0, None, "counter")
                self.tik_instance.vreduce(self.n_fft, tmp_ub_2, tensor, 2, 1, 1, 8, 0, 0, None, "counter")

                self.tik_instance.data_move(tensor, tmp_ub_1, 0, nburst, burst, src_stride, dst_stride)
                self.tik_instance.data_move(tensor[burst_elem:], tmp_ub_2, 0, nburst, burst, src_stride, dst_stride)
    
    
    def _permute_fft_implementation(self, tensor, custom_masks_ub = None):
        if self.pure_power_of_two:
            self._permute_fft_implementation_p2(tensor)
            
        elif self.n_fft == 400:
            self._permute_fft_implementation_400(tensor, custom_masks_ub)

    def _permute_fft(self, data_real_ub, data_imag_ub, custom_masks_ub = None):
        self._permute_fft_implementation(data_real_ub, custom_masks_ub)
        self._permute_fft_implementation(data_imag_ub, custom_masks_ub)

    def _dft_8(self, dft_count, data_real_ub, data_imag_ub, coeff_data_real_ub, coeff_data_imag_ub):
        with self.tik_instance.new_stmt_scope():
            tmp_real_ub = self.tik_instance.Tensor(dtype="float32", shape=(dft_count * 64,), name="tmp_real_ub", scope=tik.scope_ubuf)
            tmp_imag_ub = self.tik_instance.Tensor(dtype="float32", shape=(dft_count * 64,), name="tmp_imag_ub", scope=tik.scope_ubuf)

            tmp_ub_1 = self.tik_instance.Tensor(dtype="float32", shape=(dft_count * 64,), name="tmp_ub_1", scope=tik.scope_ubuf)
            tmp_ub_2 = self.tik_instance.Tensor(dtype="float32", shape=(dft_count * 64,), name="tmp_ub_2", scope=tik.scope_ubuf)
            tmp_ub_3 = self.tik_instance.Tensor(dtype="float32", shape=(dft_count * 8,), name="tmp_ub_3", scope=tik.scope_ubuf)

            mask = 64
            repeat_times = dft_count * 64 // mask

            vcadd_max_repeat = (dft_count * 8) // 224
            vcadd_remain_repeat = (dft_count * 8) % 224

            for i in range(8):
                self.tik_instance.data_move(tmp_real_ub[i * 8], data_real_ub, 0, dft_count, 1, 0, 7)
                self.tik_instance.data_move(tmp_imag_ub[i * 8], data_imag_ub, 0, dft_count, 1, 0, 7)

            self.tik_instance.vec_mul(mask, tmp_ub_1, tmp_real_ub, coeff_data_real_ub, repeat_times, 8, 8, 0)
            self.tik_instance.vec_mul(mask, tmp_ub_2, tmp_imag_ub, coeff_data_imag_ub, repeat_times, 8, 8, 0)
            self.tik_instance.vec_sub(mask, tmp_ub_2, tmp_ub_1, tmp_ub_2, repeat_times, 8, 8, 8)
            
            for i in range(vcadd_max_repeat):
                src_offset = i * 224 * 8
                dst_offset = i * 224
                self.tik_instance.vcadd(8, tmp_ub_3[dst_offset], tmp_ub_2[src_offset], 224, 1, 1, 1)
            src_offset = vcadd_max_repeat * 224 * 8
            dst_offset = vcadd_max_repeat * 224
            self.tik_instance.vcadd(8, tmp_ub_3[dst_offset], tmp_ub_2[src_offset], vcadd_remain_repeat, 1, 1, 1)
            self.tik_instance.data_move(data_real_ub, tmp_ub_3, 0, 1, dft_count * 8 * 4 // 32, 0, 0)

            self.tik_instance.vec_mul(mask, tmp_ub_1, tmp_real_ub, coeff_data_imag_ub, repeat_times, 8, 8, 0)
            self.tik_instance.vec_mul(mask, tmp_ub_2, tmp_imag_ub, coeff_data_real_ub, repeat_times, 8, 8, 0)
            self.tik_instance.vec_add(mask, tmp_ub_2, tmp_ub_1, tmp_ub_2, repeat_times, 8, 8, 8)
            
            for i in range(vcadd_max_repeat):
                src_offset = i * 224 * 8
                dst_offset = i * 224
                self.tik_instance.vcadd(8, tmp_ub_3[dst_offset], tmp_ub_2[src_offset], 224, 1, 1, 1)
            src_offset = vcadd_max_repeat * 224 * 8
            dst_offset = vcadd_max_repeat * 224
            self.tik_instance.vcadd(8, tmp_ub_3[dst_offset], tmp_ub_2[src_offset], vcadd_remain_repeat, 1, 1, 1)
            self.tik_instance.data_move(data_imag_ub, tmp_ub_3, 0, 1, dft_count * 8 * 4 // 32, 0, 0)

    def twiddle_mul(self, data_real_ub, data_imag_ub, coeff_data_real_ub, coeff_data_imag_ub, n_merge, factor):
        if (not self.ascend910b and self.n_fft < 64) or (self.ascend910b and self.n_fft != 400 and self.n_fft <= 1024):
            elements_per_operation = n_merge - n_merge // factor
            mask = min(64, elements_per_operation)
            repeat_times = max(1, self.n_fft // n_merge)
            repeat_mask = max(1, elements_per_operation // 64)
            mask_left_over = elements_per_operation % 64 if elements_per_operation > 64 else 0

            tmp_ub_1 = self.tik_instance.Tensor(dtype="float32", shape=(repeat_times, elements_per_operation,), name="tmp_ub_1", scope=tik.scope_ubuf)
            tmp_ub_2 = self.tik_instance.Tensor(dtype="float32", shape=(repeat_times, elements_per_operation,), name="tmp_ub_2", scope=tik.scope_ubuf)
            tmp_ub_3 = self.tik_instance.Tensor(dtype="float32", shape=(repeat_times, elements_per_operation,), name="tmp_ub_2", scope=tik.scope_ubuf)

            dst_stride = elements_per_operation * 4 // 32
            src_1_stride = dst_stride + ((n_merge // factor) * 4 // 32)

            for i in range(repeat_mask):
                self.tik_instance.vec_mul(mask, tmp_ub_1[i * 64], coeff_data_real_ub[i * 64], data_real_ub[i * 64 + n_merge // factor:], repeat_times, dst_stride, 0, src_1_stride)
                self.tik_instance.vec_mul(mask, tmp_ub_2[i * 64], coeff_data_real_ub[i * 64], data_imag_ub[i * 64 + n_merge // factor:], repeat_times, dst_stride, 0, src_1_stride)
                self.tik_instance.vec_mul(mask, tmp_ub_3[i * 64], coeff_data_imag_ub[i * 64], data_real_ub[i * 64 + n_merge // factor:], repeat_times, dst_stride, 0, src_1_stride)
                self.tik_instance.vec_mul(mask, data_imag_ub[i * 64 + n_merge // factor:], coeff_data_imag_ub[i * 64], data_imag_ub[i * 64 + n_merge // factor:], 
                                          repeat_times, src_1_stride, 0, src_1_stride)

                self.tik_instance.vec_sub(mask, data_real_ub[i * 64 + n_merge // factor:], tmp_ub_1[i * 64], data_imag_ub[i * 64 + n_merge // factor:], repeat_times, src_1_stride, dst_stride, src_1_stride)
                self.tik_instance.vec_add(mask, data_imag_ub[i * 64 + n_merge // factor:], tmp_ub_2[i * 64], tmp_ub_3[i * 64], repeat_times, src_1_stride, dst_stride, dst_stride)

            if mask_left_over:
                left_over_offset = repeat_mask * 64
                self.tik_instance.vec_mul(mask_left_over, tmp_ub_1[left_over_offset], coeff_data_real_ub[left_over_offset], data_real_ub[left_over_offset + n_merge // factor:], 1, 0, 0, 0)
                self.tik_instance.vec_mul(mask_left_over, tmp_ub_2[left_over_offset], coeff_data_real_ub[left_over_offset], data_imag_ub[left_over_offset + n_merge // factor:], 1, 0, 0, 0)
                self.tik_instance.vec_mul(mask_left_over, tmp_ub_3[left_over_offset], coeff_data_imag_ub[left_over_offset], data_real_ub[left_over_offset + n_merge // factor:], 1, 0, 0, 0)
                self.tik_instance.vec_mul(mask_left_over, data_imag_ub[left_over_offset + n_merge // factor:], coeff_data_imag_ub[left_over_offset], 
                                          data_imag_ub[left_over_offset + n_merge // factor:], 1, 0, 0, 0)

                self.tik_instance.vec_sub(mask_left_over, data_real_ub[left_over_offset + n_merge // factor:], tmp_ub_1[left_over_offset], data_imag_ub[left_over_offset + n_merge // factor:], 1, 0, 0, 0)
                self.tik_instance.vec_add(mask_left_over, data_imag_ub[left_over_offset + n_merge // factor:], tmp_ub_2[left_over_offset], tmp_ub_3[left_over_offset], 1, 0, 0, 0)
        else:
            tmp_ub_1 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge - n_merge // factor,), name="tmp_ub_1", scope=tik.scope_ubuf)
            tmp_ub_2 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge - n_merge // factor,), name="tmp_ub_2", scope=tik.scope_ubuf)
            tmp_ub_3 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge - n_merge // factor,), name="tmp_ub_3", scope=tik.scope_ubuf)
            tmp_ub_4 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge - n_merge // factor,), name="tmp_ub_4", scope=tik.scope_ubuf)

            mask = min(64, n_merge - n_merge // factor)
            repeat_times = max(1, (n_merge - n_merge // factor) // 64)
            last_num = (n_merge - n_merge // factor) % 64 if repeat_times > 1 else 0
            offset =  mask * repeat_times
            for i in range(self.n_fft // n_merge):
                self.tik_instance.vec_mul(mask, tmp_ub_1, coeff_data_real_ub, data_real_ub[i * n_merge + n_merge // factor: i * n_merge + n_merge], repeat_times, 8, 8, 8)
                self.tik_instance.vec_mul(mask, tmp_ub_2, coeff_data_imag_ub, data_imag_ub[i * n_merge + n_merge // factor: i * n_merge + n_merge], repeat_times, 8, 8, 8)
                self.tik_instance.vec_mul(mask, tmp_ub_3, coeff_data_real_ub, data_imag_ub[i * n_merge + n_merge // factor: i * n_merge + n_merge], repeat_times, 8, 8, 8)
                self.tik_instance.vec_mul(mask, tmp_ub_4, coeff_data_imag_ub, data_real_ub[i * n_merge + n_merge // factor: i * n_merge + n_merge], repeat_times, 8, 8, 8)
                self.tik_instance.vec_sub(mask, data_real_ub[i * n_merge + n_merge // factor: i * n_merge + n_merge], tmp_ub_1, tmp_ub_2, repeat_times, 8, 8, 8)
                self.tik_instance.vec_add(mask, data_imag_ub[i * n_merge + n_merge // factor: i * n_merge + n_merge], tmp_ub_3, tmp_ub_4, repeat_times, 8, 8, 8)
                if last_num > 0:
                    self.tik_instance.vec_mul(last_num, tmp_ub_1[offset], coeff_data_real_ub[offset], data_real_ub[i * n_merge + n_merge // factor + offset: i * n_merge + n_merge], 1, 8, 8, 8)
                    self.tik_instance.vec_mul(last_num, tmp_ub_2[offset], coeff_data_imag_ub[offset], data_imag_ub[i * n_merge + n_merge // factor + offset: i * n_merge + n_merge], 1, 8, 8, 8)
                    self.tik_instance.vec_mul(last_num, tmp_ub_3[offset], coeff_data_real_ub[offset], data_imag_ub[i * n_merge + n_merge // factor + offset: i * n_merge + n_merge], 1, 8, 8, 8)
                    self.tik_instance.vec_mul(last_num, tmp_ub_4[offset], coeff_data_imag_ub[offset], data_real_ub[i * n_merge + n_merge // factor + offset: i * n_merge + n_merge], 1, 8, 8, 8)
                    self.tik_instance.vec_sub(last_num, data_real_ub[i * n_merge + n_merge // factor + offset: i * n_merge + n_merge], tmp_ub_1[offset], tmp_ub_2[offset], 1, 8, 8, 8)
                    self.tik_instance.vec_add(last_num, data_imag_ub[i * n_merge + n_merge // factor + offset: i * n_merge + n_merge], tmp_ub_3[offset], tmp_ub_4[offset], 1, 8, 8, 8)

    def fft_merge_2(self, n_merge, x1_real_ub, x1_imag_ub, x2_real_ub, x2_imag_ub, y_real_ub, y_imag_ub, z_real_ub, z_imag_ub):
        with self.tik_instance.new_stmt_scope():
            tmp_ub_1 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 2,), name="tmp_ub_1", scope=tik.scope_ubuf)
            tmp_ub_2 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 2,), name="tmp_ub_2", scope=tik.scope_ubuf)

            mask = min(64, n_merge // 2)
            repeat_times = max(1, (n_merge // 2) // 64)
            last_num = (n_merge // 2) % 64 if repeat_times > 1 else 0
            offset =  mask * repeat_times

            #Xk+N/2
            self.tik_instance.vec_sub(mask, tmp_ub_1, y_real_ub, z_real_ub, repeat_times, 8, 8, 8)
            self.tik_instance.vec_sub(mask, tmp_ub_2, y_imag_ub, z_imag_ub, repeat_times, 8, 8, 8)
            if last_num > 0:
                self.tik_instance.vec_sub(last_num, tmp_ub_1[offset], y_real_ub[offset], z_real_ub[offset], 1, 8, 8, 8)
                self.tik_instance.vec_sub(last_num, tmp_ub_2[offset], y_imag_ub[offset], z_imag_ub[offset], 1, 8, 8, 8)

            #Xk
            self.tik_instance.vec_add(mask, x1_real_ub, y_real_ub, z_real_ub, repeat_times, 8, 8, 8)
            self.tik_instance.vec_add(mask, x1_imag_ub, y_imag_ub, z_imag_ub, repeat_times, 8, 8, 8)
            if last_num > 0:
                self.tik_instance.vec_add(last_num, x1_real_ub[offset], y_real_ub[offset], z_real_ub[offset], 1, 8, 8, 8)
                self.tik_instance.vec_add(last_num, x1_imag_ub[offset], y_imag_ub[offset], z_imag_ub[offset], 1, 8, 8, 8)

            self.tik_instance.data_move(x2_real_ub, tmp_ub_1, 0, 1, n_merge // 2 * 4 // 32, 0, 0)
            self.tik_instance.data_move(x2_imag_ub, tmp_ub_2, 0, 1, n_merge // 2 * 4 // 32, 0, 0)

    def fft_merge_4(self, n_merge, x1_real_ub, x1_imag_ub,
                        x2_real_ub, x2_imag_ub,
                        x3_real_ub, x3_imag_ub,
                        x4_real_ub, x4_imag_ub,
                        y_real_ub, y_imag_ub,
                        z_real_ub, z_imag_ub,
                        g_real_ub, g_imag_ub,
                        h_real_ub, h_imag_ub):
        with self.tik_instance.new_stmt_scope():
            tmp_ub_1 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 4,), name="tmp_ub_1", scope=tik.scope_ubuf)
            tmp_ub_2 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 4,), name="tmp_ub_2", scope=tik.scope_ubuf)
            tmp_ub_3 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 4,), name="tmp_ub_3", scope=tik.scope_ubuf)
            tmp_ub_4 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 4,), name="tmp_ub_4", scope=tik.scope_ubuf)
            tmp_ub_5 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 4,), name="tmp_ub_5", scope=tik.scope_ubuf)
            tmp_ub_6 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 4,), name="tmp_ub_6", scope=tik.scope_ubuf)
            tmp_ub_7 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 4,), name="tmp_ub_7", scope=tik.scope_ubuf)
            tmp_ub_8 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 4,), name="tmp_ub_8", scope=tik.scope_ubuf)

            mask = min(64, n_merge // 4)
            repeat_times = max(1, (n_merge // 4) // 64)
            last_num = (n_merge // 4) % 64 if repeat_times > 1 else 0
            offset =  mask * repeat_times

            self.fft_merge_2(n_merge // 2, tmp_ub_1, tmp_ub_2, tmp_ub_3, tmp_ub_4, y_real_ub, y_imag_ub, g_real_ub, g_imag_ub)
            self.fft_merge_2(n_merge // 2, tmp_ub_5, tmp_ub_6, tmp_ub_7, tmp_ub_8, z_real_ub, z_imag_ub, h_real_ub, h_imag_ub)

            self.tik_instance.vmuls(mask, tmp_ub_8, tmp_ub_8, -1, repeat_times, 1, 1, 8, 8)
            if last_num > 0:
                self.tik_instance.vmuls(last_num, tmp_ub_8[offset], tmp_ub_8[offset], -1, 1, 1, 1, 8, 8)

            self.fft_merge_2(n_merge // 2, x1_real_ub, x1_imag_ub, x3_real_ub, x3_imag_ub, tmp_ub_1, tmp_ub_2, tmp_ub_5, tmp_ub_6)
            self.fft_merge_2(n_merge // 2, x4_real_ub, x4_imag_ub, x2_real_ub, x2_imag_ub, tmp_ub_3, tmp_ub_4, tmp_ub_8, tmp_ub_7)

    def fft_merge_5(self, n_merge, x1_real_ub, x1_imag_ub,
                    x2_real_ub, x2_imag_ub,
                    x3_real_ub, x3_imag_ub,
                    x4_real_ub, x4_imag_ub,
                    x5_real_ub, x5_imag_ub,
                    y_real_ub, y_imag_ub,
                    z_real_ub, z_imag_ub,
                    g_real_ub, g_imag_ub,
                    h_real_ub, h_imag_ub,
                    i_real_ub, i_imag_ub):
        ra = 0.30901699437494745
        ia = -0.9510565162951535
        rb = -0.8090169943749473
        ib = -0.5877852522924732
        with self.tik_instance.new_stmt_scope():
            tmp_ub_1 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_1", scope=tik.scope_ubuf)
            tmp_ub_2 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_2", scope=tik.scope_ubuf)
            tmp_ub_3 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_3", scope=tik.scope_ubuf)
            tmp_ub_4 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_4", scope=tik.scope_ubuf)
            tmp_ub_5 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_5", scope=tik.scope_ubuf)
            tmp_ub_6 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_6", scope=tik.scope_ubuf)
            tmp_ub_7 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_7", scope=tik.scope_ubuf)
            tmp_ub_8 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_8", scope=tik.scope_ubuf)

            tmp_ub_9 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_9", scope=tik.scope_ubuf)
            tmp_ub_10 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_10", scope=tik.scope_ubuf)
            tmp_ub_11 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_11", scope=tik.scope_ubuf)
            tmp_ub_12 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_12", scope=tik.scope_ubuf)
            tmp_ub_13 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_13", scope=tik.scope_ubuf)
            tmp_ub_14 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_14", scope=tik.scope_ubuf)
            tmp_ub_15 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_15", scope=tik.scope_ubuf)
            tmp_ub_16 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge // 5,), name="tmp_ub_16", scope=tik.scope_ubuf)

            mask = min(64, n_merge // 5)
            repeat_times = max(1, (n_merge // 5) // 64)
            last_num = (n_merge // 5) % 64 if repeat_times > 1 else 0
            offset =  mask * repeat_times

            self.fft_merge_2(n_merge // 5 * 2, tmp_ub_1, tmp_ub_2, tmp_ub_3, tmp_ub_4, z_real_ub, z_imag_ub, i_real_ub, i_imag_ub)
            self.fft_merge_2(n_merge // 5 * 2, tmp_ub_5, tmp_ub_6, tmp_ub_7, tmp_ub_8, g_real_ub, g_imag_ub, h_real_ub, h_imag_ub)

            #Xk+N/5 and Xk+4N/5
            self.tik_instance.vmuls(mask, tmp_ub_9, tmp_ub_1, ra, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmuls(mask, tmp_ub_10, tmp_ub_2, ra, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmuls(mask, tmp_ub_11, tmp_ub_4, -ia, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmuls(mask, tmp_ub_12, tmp_ub_3, ia, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmuls(mask, tmp_ub_13, tmp_ub_5, rb, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmuls(mask, tmp_ub_14, tmp_ub_6, rb, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmuls(mask, tmp_ub_15, tmp_ub_8, -ib, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmuls(mask, tmp_ub_16, tmp_ub_7, ib, repeat_times, 1, 1, 8, 8)

            self.tik_instance.vec_add(mask, tmp_ub_9, tmp_ub_9, tmp_ub_13, repeat_times, 8, 8, 8)
            self.tik_instance.vec_add(mask, tmp_ub_10, tmp_ub_10, tmp_ub_14, repeat_times, 8, 8, 8)
            self.tik_instance.vec_add(mask, tmp_ub_9, y_real_ub, tmp_ub_9, repeat_times, 8, 8, 8)
            self.tik_instance.vec_add(mask, tmp_ub_10, y_imag_ub, tmp_ub_10, repeat_times, 8, 8, 8)

            self.tik_instance.vec_add(mask, tmp_ub_11, tmp_ub_11, tmp_ub_15, repeat_times, 8, 8, 8)
            self.tik_instance.vec_add(mask, tmp_ub_12, tmp_ub_12, tmp_ub_16, repeat_times, 8, 8, 8)
            if last_num > 0:
                self.tik_instance.vmuls(last_num, tmp_ub_9[offset], tmp_ub_1[offset], ra, 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(last_num, tmp_ub_10[offset], tmp_ub_2[offset], ra, 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(last_num, tmp_ub_11[offset], tmp_ub_4[offset], -ia, 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(last_num, tmp_ub_12[offset], tmp_ub_3[offset], ia, 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(last_num, tmp_ub_13[offset], tmp_ub_5[offset], rb, 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(last_num, tmp_ub_14[offset], tmp_ub_6[offset], rb, 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(last_num, tmp_ub_15[offset], tmp_ub_8[offset], -ib, 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(last_num, tmp_ub_16[offset], tmp_ub_7[offset], ib, 1, 1, 1, 8, 8)

                self.tik_instance.vec_add(last_num, tmp_ub_9[offset], tmp_ub_9[offset], tmp_ub_13[offset], 1, 8, 8, 8)
                self.tik_instance.vec_add(last_num, tmp_ub_10[offset], tmp_ub_10[offset], tmp_ub_14[offset], 1, 8, 8, 8)
                self.tik_instance.vec_add(last_num, tmp_ub_9[offset], y_real_ub[offset], tmp_ub_9[offset], 1, 8, 8, 8)
                self.tik_instance.vec_add(last_num, tmp_ub_10[offset], y_imag_ub[offset], tmp_ub_10[offset], 1, 8, 8, 8)

                self.tik_instance.vec_add(last_num, tmp_ub_11[offset], tmp_ub_11[offset], tmp_ub_15[offset], 1, 8, 8, 8)
                self.tik_instance.vec_add(last_num, tmp_ub_12[offset], tmp_ub_12[offset], tmp_ub_16[offset], 1, 8, 8, 8)

            self.fft_merge_2(n_merge // 5 * 2, x2_real_ub, x2_imag_ub, x5_real_ub, x5_imag_ub, tmp_ub_9, tmp_ub_10, tmp_ub_11, tmp_ub_12)

            #Xk+2N/5 and Xk+3N/5
            self.tik_instance.vmuls(mask, tmp_ub_9, tmp_ub_1, rb, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmuls(mask, tmp_ub_10, tmp_ub_2, rb, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmuls(mask, tmp_ub_11, tmp_ub_4, -ib, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmuls(mask, tmp_ub_12, tmp_ub_3, ib, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmuls(mask, tmp_ub_13, tmp_ub_5, ra, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmuls(mask, tmp_ub_14, tmp_ub_6, ra, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmuls(mask, tmp_ub_15, tmp_ub_8, -ia, repeat_times, 1, 1, 8, 8)
            self.tik_instance.vmuls(mask, tmp_ub_16, tmp_ub_7, ia, repeat_times, 1, 1, 8, 8)

            self.tik_instance.vec_add(mask, tmp_ub_9, tmp_ub_9, tmp_ub_13, repeat_times, 8, 8, 8)
            self.tik_instance.vec_add(mask, tmp_ub_10, tmp_ub_10, tmp_ub_14, repeat_times, 8, 8, 8)
            self.tik_instance.vec_add(mask, tmp_ub_9, y_real_ub, tmp_ub_9, repeat_times, 8, 8, 8)
            self.tik_instance.vec_add(mask, tmp_ub_10, y_imag_ub, tmp_ub_10, repeat_times, 8, 8, 8)

            self.tik_instance.vec_sub(mask, tmp_ub_11, tmp_ub_11, tmp_ub_15, repeat_times, 8, 8, 8)
            self.tik_instance.vec_sub(mask, tmp_ub_12, tmp_ub_12, tmp_ub_16, repeat_times, 8, 8, 8)

            if last_num > 0:
                self.tik_instance.vmuls(last_num, tmp_ub_9[offset], tmp_ub_1[offset], rb, 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(last_num, tmp_ub_10[offset], tmp_ub_2[offset], rb, 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(last_num, tmp_ub_11[offset], tmp_ub_4[offset], -ib, 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(last_num, tmp_ub_12[offset], tmp_ub_3[offset], ib, 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(last_num, tmp_ub_13[offset], tmp_ub_5[offset], ra, 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(last_num, tmp_ub_14[offset], tmp_ub_6[offset], ra, 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(last_num, tmp_ub_15[offset], tmp_ub_8[offset], -ia, 1, 1, 1, 8, 8)
                self.tik_instance.vmuls(last_num, tmp_ub_16[offset], tmp_ub_7[offset], ia, 1, 1, 1, 8, 8)

                self.tik_instance.vec_add(last_num, tmp_ub_9[offset], tmp_ub_9[offset], tmp_ub_13[offset], 1, 8, 8, 8)
                self.tik_instance.vec_add(last_num, tmp_ub_10[offset], tmp_ub_10[offset], tmp_ub_14[offset], 1, 8, 8, 8)
                self.tik_instance.vec_add(last_num, tmp_ub_9[offset], y_real_ub[offset], tmp_ub_9[offset], 1, 8, 8, 8)
                self.tik_instance.vec_add(last_num, tmp_ub_10[offset], y_imag_ub[offset], tmp_ub_10[offset], 1, 8, 8, 8)

                self.tik_instance.vec_sub(last_num, tmp_ub_11[offset], tmp_ub_11[offset], tmp_ub_15[offset], 1, 8, 8, 8)
                self.tik_instance.vec_sub(last_num, tmp_ub_12[offset], tmp_ub_12[offset], tmp_ub_16[offset], 1, 8, 8, 8)

            self.fft_merge_2(n_merge // 5 * 2, x3_real_ub, x3_imag_ub, x4_real_ub, x4_imag_ub, tmp_ub_9, tmp_ub_10, tmp_ub_11, tmp_ub_12)

            #Xk
            self.tik_instance.vec_add(mask, tmp_ub_1, tmp_ub_1, tmp_ub_5, repeat_times, 8, 8, 8)
            self.tik_instance.vec_add(mask, tmp_ub_2, tmp_ub_2, tmp_ub_6, repeat_times, 8, 8, 8)
            self.tik_instance.vec_add(mask, x1_real_ub, y_real_ub, tmp_ub_1, repeat_times, 8, 8, 8)
            self.tik_instance.vec_add(mask, x1_imag_ub, y_imag_ub, tmp_ub_2, repeat_times, 8, 8, 8)
            if last_num > 0:
                self.tik_instance.vec_add(last_num, tmp_ub_1[offset], tmp_ub_1[offset], tmp_ub_5[offset], 1, 8, 8, 8)
                self.tik_instance.vec_add(last_num, tmp_ub_2[offset], tmp_ub_2[offset], tmp_ub_6[offset], 1, 8, 8, 8)
                self.tik_instance.vec_add(last_num, x1_real_ub[offset], y_real_ub[offset], tmp_ub_1[offset], 1, 8, 8, 8)
                self.tik_instance.vec_add(last_num, x1_imag_ub[offset], y_imag_ub[offset], tmp_ub_2[offset], 1, 8, 8, 8)

    def _fft1d(self, data_real_ub, data_imag_ub, coeff_data_real_ub, coeff_data_imag_ub, custom_masks_ub = None):
        self._permute_fft(data_real_ub, data_imag_ub, custom_masks_ub)
        
        repeat_dft = max(self.n_fft // 512, 1)
        dft_count = min(64, self.n_fft // 8)
        if self.n_fft == 400:
            repeat_dft = 1
            dft_count = self.n_fft // 8
        
        with self.tik_instance.new_stmt_scope():
            if self.n_fft >= 4096:
                coeff_data_real_ub1 = self.tik_instance.Tensor(dtype="float32", shape=(64,), name="coeff_data_real_ub1", scope=tik.scope_ubuf)
                coeff_data_imag_ub1 = self.tik_instance.Tensor(dtype="float32", shape=(64,), name="coeff_data_imag_ub1", scope=tik.scope_ubuf)

                self.tik_instance.data_move(coeff_data_real_ub1, self.coeff_data_real, 0, 1, 64 * 4 // 32, 0, 0)
                self.tik_instance.data_move(coeff_data_imag_ub1, self.coeff_data_imag, 0, 1, 64 * 4 // 32, 0, 0)

                for i in range(repeat_dft):
                    self._dft_8(dft_count, data_real_ub[i * dft_count * 8: i * dft_count * 8 + dft_count * 8],
                                data_imag_ub[i * dft_count * 8: i * dft_count * 8 + dft_count * 8],
                                coeff_data_real_ub1, coeff_data_imag_ub1)

            else:
                for i in range(repeat_dft):      
                    self._dft_8(dft_count, data_real_ub[i * dft_count * 8: i * dft_count * 8 + dft_count * 8],
                                data_imag_ub[i * dft_count * 8: i * dft_count * 8 + dft_count * 8],
                                coeff_data_real_ub[0: 64], coeff_data_imag_ub[0: 64])
                    
        coeff_index = 64
        for i in range(1, len(self.factors)):
            factor = self.factors[i]
            n_merge = reduce(lambda x, y: x * y, self.factors[:i + 1])
            with self.tik_instance.new_stmt_scope():
                if self.n_fft >= 4096:
                    coeff_data_real_ub2 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge,), name="coeff_data_real_ub2", scope=tik.scope_ubuf)
                    coeff_data_imag_ub2 = self.tik_instance.Tensor(dtype="float32", shape=(n_merge,), name="coeff_data_imag_ub2", scope=tik.scope_ubuf)

                    self.tik_instance.data_move(coeff_data_real_ub2, self.coeff_data_real[coeff_index: coeff_index + (n_merge - n_merge // factor)], 0, 1, n_merge * 4 // 32, 0, 0)
                    self.tik_instance.data_move(coeff_data_imag_ub2, self.coeff_data_imag[coeff_index: coeff_index + (n_merge - n_merge // factor)], 0, 1, n_merge * 4 // 32, 0, 0)
                    
                    self.twiddle_mul(data_real_ub, data_imag_ub, coeff_data_real_ub2, coeff_data_imag_ub2, n_merge, factor)
                else:
                    self.twiddle_mul(data_real_ub, data_imag_ub, coeff_data_real_ub[coeff_index: coeff_index + (n_merge - n_merge // factor)], coeff_data_imag_ub[coeff_index: coeff_index + (n_merge - n_merge // factor)], n_merge, factor)
                    
            for j in range(self.n_fft // n_merge):
                if factor == 2:
                    self.fft_merge_2(n_merge, data_real_ub[j * n_merge: j * n_merge + n_merge // 2],
                                                data_imag_ub[j * n_merge: j * n_merge + n_merge // 2],
                                                data_real_ub[j * n_merge + n_merge // 2: j * n_merge + n_merge],
                                                data_imag_ub[j * n_merge + n_merge // 2: j * n_merge + n_merge],
                                                data_real_ub[j * n_merge: j * n_merge + n_merge // 2],
                                                data_imag_ub[j * n_merge: j * n_merge + n_merge // 2],
                                                data_real_ub[j * n_merge + n_merge // 2: j * n_merge + n_merge],
                                                data_imag_ub[j * n_merge + n_merge // 2: j * n_merge + n_merge])
                elif factor == 4:
                    self.fft_merge_4(n_merge, data_real_ub[j * n_merge: j * n_merge + n_merge // 4],
                                                data_imag_ub[j * n_merge: j * n_merge + n_merge // 4],
                                                data_real_ub[j * n_merge + n_merge // 4: j * n_merge + 2 * n_merge // 4],
                                                data_imag_ub[j * n_merge + n_merge // 4: j * n_merge + 2 * n_merge // 4],
                                                data_real_ub[j * n_merge + 2 * n_merge // 4: j * n_merge + 3 * n_merge // 4],
                                                data_imag_ub[j * n_merge + 2 * n_merge // 4: j * n_merge + 3 * n_merge // 4],
                                                data_real_ub[j * n_merge + 3 * n_merge // 4: j * n_merge + n_merge],
                                                data_imag_ub[j * n_merge + 3 * n_merge // 4: j * n_merge + n_merge],
                                                data_real_ub[j * n_merge: j * n_merge + n_merge // 4],
                                                data_imag_ub[j * n_merge: j * n_merge + n_merge // 4],
                                                data_real_ub[j * n_merge + n_merge // 4: j * n_merge + 2 * n_merge // 4],
                                                data_imag_ub[j * n_merge + n_merge // 4: j * n_merge + 2 * n_merge // 4],
                                                data_real_ub[j * n_merge + 2 * n_merge // 4: j * n_merge + 3 * n_merge // 4],
                                                data_imag_ub[j * n_merge + 2 * n_merge // 4: j * n_merge + 3 * n_merge // 4],
                                                data_real_ub[j * n_merge + 3 * n_merge // 4: j * n_merge + n_merge],
                                                data_imag_ub[j * n_merge + 3 * n_merge // 4: j * n_merge + n_merge])
                elif factor == 5:
                    self.fft_merge_5(n_merge, data_real_ub[j * n_merge: j * n_merge + n_merge // 5],
                                                data_imag_ub[j * n_merge: j * n_merge + n_merge // 5],
                                                data_real_ub[j * n_merge + n_merge // 5: j * n_merge + 2 * n_merge // 5],
                                                data_imag_ub[j * n_merge + n_merge // 5: j * n_merge + 2 * n_merge // 5],
                                                data_real_ub[j * n_merge + 2 * n_merge // 5: j * n_merge + 3 * n_merge // 5],
                                                data_imag_ub[j * n_merge + 2 * n_merge // 5: j * n_merge + 3 * n_merge // 5],
                                                data_real_ub[j * n_merge + 3 * n_merge // 5: j * n_merge + 4 * n_merge // 5],
                                                data_imag_ub[j * n_merge + 3 * n_merge // 5: j * n_merge + 4 * n_merge // 5],
                                                data_real_ub[j * n_merge + 4 * n_merge // 5: j * n_merge + n_merge],
                                                data_imag_ub[j * n_merge + 4 * n_merge // 5: j * n_merge + n_merge],
                                                data_real_ub[j * n_merge: j * n_merge + n_merge // 5],
                                                data_imag_ub[j * n_merge: j * n_merge + n_merge // 5],
                                                data_real_ub[j * n_merge + n_merge // 5: j * n_merge + 2 * n_merge // 5],
                                                data_imag_ub[j * n_merge + n_merge // 5: j * n_merge + 2 * n_merge // 5],
                                                data_real_ub[j * n_merge + 2 * n_merge // 5: j * n_merge + 3 * n_merge // 5],
                                                data_imag_ub[j * n_merge + 2 * n_merge // 5: j * n_merge + 3 * n_merge // 5],
                                                data_real_ub[j * n_merge + 3 * n_merge // 5: j * n_merge + 4 * n_merge // 5],
                                                data_imag_ub[j * n_merge + 3 * n_merge // 5: j * n_merge + 4 * n_merge // 5],
                                                data_real_ub[j * n_merge + 4 * n_merge // 5: j * n_merge + n_merge],
                                                data_imag_ub[j * n_merge + 4 * n_merge // 5: j * n_merge + n_merge])
            coeff_index += n_merge - n_merge // factor

    def _window_mul(self, data_real_ub, data_imag_ub, window_data_ub):
        with self.tik_instance.new_stmt_scope():
            if self.n_fft >= 4096:
                window_data_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft,), name="window_data_ub", scope=tik.scope_ubuf)
    
            self.tik_instance.vec_dup(self.mask, window_data_ub, 0, self.vec_repeat_times, 8)
            if self.mask_leftover:
                self.tik_instance.vec_dup(self.mask_leftover, window_data_ub[self.elems_multiplyed], 0, 1, 8)
            self.tik_instance.data_move(window_data_ub[(self.n_fft - self.win_length) // 2], self.window_data, 0, 1, self.win_length * 4 // 32, 0, 0)
    
            self.tik_instance.vec_mul(self.mask, data_real_ub, window_data_ub, data_real_ub, self.vec_repeat_times, 8, 8, 8)
            self.tik_instance.vec_mul(self.mask, data_imag_ub, window_data_ub, data_imag_ub, self.vec_repeat_times, 8, 8, 8)
    
            if self.mask_leftover:
                self.tik_instance.vec_mul(self.mask_leftover, data_real_ub[self.elems_multiplyed:], window_data_ub[self.elems_multiplyed:], 
                                          data_real_ub[self.elems_multiplyed:], 1, 8, 8, 8)
                self.tik_instance.vec_mul(self.mask_leftover, data_imag_ub[self.elems_multiplyed:], window_data_ub[self.elems_multiplyed:], 
                                          data_imag_ub[self.elems_multiplyed:], 1, 8, 8, 8)

    def _transpose_and_move(self, frame_i, batch_i, block_transpose_dst_offset):
        self._transpose()
        self._move_transposed_data(frame_i, batch_i, block_transpose_dst_offset)

    def _transpose(self):
        repeat_vec_trans_times = max(2, self.second_elements // 8)
        
        dst_list = [self.transpose_data_output_ub[8 * i] for i in range(16)]
        src_list = [self.transpose_data_input_ub[self.second_elements * i] for i in range(16)]
        
        self.tik_instance.vec_trans_scatter(False, False, dst_list, src_list, repeat_vec_trans_times, 16, 1)

    def _move_transposed_data(self, frame_i, batch_i, block_transpose_dst_offset):
        batch_offset = batch_i * self.n_fft * self.frame_count * 2
        is_last_frame = frame_i == (self.frame_count - 1)
        
        frame_count_box_8 = self.frame_count // 8
        frame_count_left_over_8 = self.frame_count % 8

        is_src_gap = self.tik_instance.Scalar(dtype="uint8", init_value=frame_i % 8 <= 3)

        if frame_count_left_over_8:
            with self.tik_instance.if_scope(is_last_frame):
                if self.ascend910b:
                    self.tik_instance.data_move_pad(self.output_tmp_data[batch_offset + block_transpose_dst_offset + 2 * (frame_i  - (frame_count_left_over_8 - 1))], 
                                                    self.transpose_data_output_ub, self.second_elements, 4 * 2 * frame_count_left_over_8 
                                                    + (not frame_count_left_over_8), 4 * 2 * 8 * frame_count_box_8, is_src_gap)
                else:
                    dst_adr = batch_offset + block_transpose_dst_offset + 2 * (frame_i  - (frame_count_left_over_8 - 1))
                    for i in range(self.second_elements):
                        dst_offset = i * (2 * frame_count_left_over_8 + 2 * 8 * frame_count_box_8)
                        src_offset = 16 * i
                        self._data_move_pad_310p(self.output_tmp_data[dst_adr + dst_offset:], 
                                                self.transpose_data_output_ub[src_offset:], 2 * frame_count_left_over_8, True)
            with self.tik_instance.else_scope():
                output_index = self.tik_instance.Scalar(dtype="int64", init_value = 2 * (frame_i - 7))
                self.tik_instance.scalar_max(output_index, 0, output_index)

                if self.ascend910b:
                    self.tik_instance.data_move_pad(self.output_tmp_data[batch_offset + block_transpose_dst_offset + output_index], self.transpose_data_output_ub, 
                                                    self.second_elements, 4 * 16, 4 * (2 * 8 * (max(0, (frame_count_box_8 - 1))) 
                                                                                + 2 * frame_count_left_over_8), 0)
                else:
                    dst_adr = batch_offset + block_transpose_dst_offset + output_index
                    for i in range(self.second_elements):
                        dst_offset = i * (16 + (2 * 8 * (max(0, (frame_count_box_8 - 1))) 
                                                + 2 * frame_count_left_over_8))
                        src_offset = 16 * i
                        self._data_move_pad_310p(self.output_tmp_data[dst_adr + dst_offset:], self.transpose_data_output_ub[src_offset:], 16, True)
        else:
            output_index = self.tik_instance.Scalar(dtype="int64", init_value = 2 * (frame_i - 7))
            self.tik_instance.scalar_max(output_index, 0, output_index)

            self.tik_instance.data_move(self.output_tmp_data[batch_offset + block_transpose_dst_offset + output_index], self.transpose_data_output_ub,
                                        0, self.second_elements, 2, 0, max(0, 2 * (frame_count_box_8 - 1)))

    def _parallel_transpose(self, cur_core, additional_transpose):
        with self.tik_instance.for_range(0, self.transpose_16_per_core + additional_transpose) as cur_i:
            cores_with_extra_transpose = self.tik_instance.Scalar(dtype="int64", init_value=self.transpose_16_left_over)
            cur_core_tmp = self.tik_instance.Scalar(dtype="int64", init_value=cur_core)
            self.tik_instance.scalar_min(cores_with_extra_transpose, cur_core_tmp, cores_with_extra_transpose)
            cores_with_normal_transpose = cur_core - cores_with_extra_transpose

            transpose_16_i = (cores_with_extra_transpose * (self.transpose_16_per_core + 1)) + (cores_with_normal_transpose * self.transpose_16_per_core) + cur_i
            batch_i = transpose_16_i // (self.transpose_16_num + self.transpose_num_left_over)
            batch_offset = batch_i * self.n_fft * self.frame_count * 2
            transpose_16_i_on_batch = transpose_16_i % (self.transpose_16_num + self.transpose_num_left_over)

            block_transpose_src_stride = (self.block_transpose_num - 1) * self.max_n_fft

            pre_transpose_index = self.tik_instance.Scalar(dtype="uint32", init_value=batch_offset + transpose_16_i_on_batch * 16 * self.n_fft)

            for block_transpose_i in range(self.block_transpose_num):
                block_transpose_src_offset = block_transpose_i * self.max_n_fft 
                block_transpose_dst_offset = block_transpose_i * self.max_n_fft * self.frame_count * 2
                
                frame_to_pass = self.tik_instance.Scalar(dtype="int64", init_value=(transpose_16_i_on_batch * 8 + 8) - 1)
                self.tik_instance.scalar_min(frame_to_pass, self.frame_count - 1, frame_to_pass)

                with self.tik_instance.if_scope(tik.all(frame_to_pass == (self.frame_count - 1), self.transpose_num_left_over)):
                    pre_transpose_index.set_as(batch_offset + self.transpose_16_num * 16 * self.n_fft)

                if self.ascend910b:
                    self.tik_instance.data_move(self.transpose_data_input_ub, self.pre_transpose_data[pre_transpose_index + block_transpose_src_offset], 
                                                0, 16, self.second_elements * 4 // 32, block_transpose_src_stride * 4 // 32, 0)
                else:
                    with self.tik_instance.if_scope(frame_to_pass != self.frame_count - 1):
                        self.tik_instance.data_move(self.transpose_data_input_ub, self.pre_transpose_data[pre_transpose_index + block_transpose_src_offset], 
                                                    0, 16, self.second_elements * 4 // 32, block_transpose_src_stride * 4 // 32, 0)
                    
                    with self.tik_instance.else_scope():
                        frames_to_read = self.tik_instance.Scalar(dtype="int64", init_value=self.frame_count % 8)
                        self.tik_instance.scalar_min(frames_to_read, frames_to_read, 8)
                        with self.tik_instance.if_scope(frames_to_read == 0):
                            frames_to_read.set_as(8)

                        self.tik_instance.data_move(self.transpose_data_input_ub, self.pre_transpose_data[pre_transpose_index + block_transpose_src_offset], 
                                                    0, frames_to_read * 2, self.second_elements * 4 // 32, block_transpose_src_stride * 4 // 32, 0)

                self._transpose_and_move(frame_to_pass, batch_i, block_transpose_dst_offset)

    def stft_compute(self):
        self.mask = min(64, self.n_fft)
        self.vec_repeat_times = max(1, self.n_fft // 64)
        self.mask_leftover = self.n_fft % 64 if self.vec_repeat_times > 1 else 0
        self.elems_multiplyed = self.mask * self.vec_repeat_times

        #data for multicore transpose
        self.transpose_16_num = self.frame_count // 8
        self.transpose_num_left_over = (self.frame_count % 8) != 0
        self.transpose_16_overall_per_batch = (self.transpose_16_num + self.transpose_num_left_over)
        self.transpose_16_overall = self.transpose_16_overall_per_batch * self.batches

        #data for multicore pad
        self.cores_for_batches = min(self.overall_frame_count, self.cores_num_910B)
        self.batch_per_core = self.batches // self.cores_for_batches
        self.left_over_batches = self.batches % self.cores_for_batches

        #data for transpose
        self.max_n_fft = 1024
        self.second_elements = min(self.n_fft, self.max_n_fft)
        self.block_transpose_num = self.n_fft // self.second_elements

        #data for pad
        self.is_additional_pad_space_needed = self.pad_mode in ["circular", "reflect"]
        self.is_reflect = self.pad_mode == "reflect"
        self.is_replicate = self.pad_mode == "replicate"
        
        custom_masks_ub = None
        
        if self.overall_frame_count < self.cores_num_910B:
            self.transpose_16_per_core = self.transpose_16_overall // self.overall_frame_count
            self.transpose_16_left_over = self.transpose_16_overall % self.overall_frame_count

            if not self.ascend910b:
                self.transpose_16_per_core = self.transpose_16_overall
                self.transpose_16_left_over = 0

            if self.onesided:
                overall_onesided_moves = self.batches * (self.n_fft // 2 + 1)
                onesided_move_per_core = overall_onesided_moves // self.overall_frame_count
                onesided_move_left_over = overall_onesided_moves % self.overall_frame_count

            with self.tik_instance.for_range(0, self.overall_frame_count, block_num = self.overall_frame_count) as frame_i:
                additional_batch = self.tik_instance.Scalar(dtype="uint8", init_value=frame_i < self.left_over_batches)

                cores_with_extra_batches = self.tik_instance.Scalar(dtype="int64", init_value=self.left_over_batches)
                cur_core = self.tik_instance.Scalar(dtype="int64", init_value=frame_i)
                self.tik_instance.scalar_min(cores_with_extra_batches, cur_core, cores_with_extra_batches)
                cores_with_normal_batches = cur_core - cores_with_extra_batches
                
                if self.center:
                    self._count_pads_parallel(additional_batch, cores_with_extra_batches, cores_with_normal_batches)

                if self.overall_frame_count != 1:
                    self.tik_instance.block_barrier(self.sync_workspace)

                with self.tik_instance.new_stmt_scope():
                    data_real_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft,), name="data_real_ub", scope=tik.scope_ubuf)
                    data_imag_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft,), name="data_imag_ub", scope=tik.scope_ubuf)
                    window_data_ub = None
                    coeff_data_real_ub = None
                    coeff_data_imag_ub = None
                    tmp_ub = None
                    if self.n_fft < 4096:
                        window_data_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft,), name="window_data_ub", scope=tik.scope_ubuf)
                        coeff_data_real_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_coeff,), name="coeff_data_real_ub", scope=tik.scope_ubuf)
                        coeff_data_imag_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_coeff,), name="coeff_data_imag_ub", scope=tik.scope_ubuf)
                        tmp_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft * 2,), name="tmp_ub", scope=tik.scope_ubuf)

                        self.tik_instance.data_move(coeff_data_real_ub, self.coeff_data_real, 0, 1, self.n_coeff * 4 // 32, 0, 0)
                        self.tik_instance.data_move(coeff_data_imag_ub, self.coeff_data_imag, 0, 1, self.n_coeff * 4 // 32, 0, 0)

                    if self.n_fft == 400:
                        custom_masks_ub = self.tik_instance.Tensor(dtype="uint32", shape=(50, 16), name="custom_masks_ub", scope=tik.scope_ubuf)
                        
                        counter = 0
                        for i in range(2):
                            for j in range(5):
                                for k in range(5):
                                    if self.ascend910b:
                                        self.tik_instance.data_move_pad(custom_masks_ub[counter * 16], self.custom_masks_400[i][j][k], 1, (416 // 32) * 4, 0, 0, 0, 0)
                                    else:
                                        self._data_move_pad_310p(custom_masks_ub[counter * 16:], self.custom_masks_400[i][j][k], 416 // 32)
                                    counter += 1

                    self._fft_frame(frame_i, data_real_ub, data_imag_ub, window_data_ub, coeff_data_real_ub, coeff_data_imag_ub, tmp_ub, custom_masks_ub)
                
                if self.overall_frame_count != 1:
                    self.tik_instance.block_barrier(self.sync_workspace)

                if self.ascend910b:
                    with self.tik_instance.if_scope(frame_i < self.transpose_16_overall):
                        additional_transpose = self.tik_instance.Scalar(dtype="uint8", init_value=frame_i < self.transpose_16_left_over)
                        self.transpose_data_input_ub = self.tik_instance.Tensor(dtype="float32", shape=(16, self.second_elements), name="transpose_data_input_ub", scope=tik.scope_ubuf)
                        self.transpose_data_output_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.second_elements, 16), name="transpose_data_output_ub", scope=tik.scope_ubuf)
                        self._parallel_transpose(frame_i, additional_transpose)           
                else:
                    with self.tik_instance.if_scope(frame_i == 0):
                        self.transpose_data_input_ub = self.tik_instance.Tensor(dtype="float32", shape=(16, self.second_elements), name="transpose_data_input_ub", scope=tik.scope_ubuf)
                        self.transpose_data_output_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.second_elements, 16), name="transpose_data_output_ub", scope=tik.scope_ubuf)
                        self._parallel_transpose(frame_i, 0)
                        self.tik_instance.data_move(self.transpose_data_output_ub, self.transpose_data_input_ub, 0, 1, 1, 0, 0)          

                if self.overall_frame_count != 1:
                    self.tik_instance.block_barrier(self.sync_workspace)

                if self.onesided:
                    if self.ascend910b:
                        with self.tik_instance.if_scope(cur_core < overall_onesided_moves):                    
                            additional_onesided_move = self.tik_instance.Scalar(dtype="uint8", init_value=cur_core < onesided_move_left_over)
                            
                            cores_with_extra_onesided = self.tik_instance.Scalar(dtype="int64", init_value=onesided_move_left_over)
                            cur_core_64 = self.tik_instance.Scalar(dtype="int64", init_value=cur_core)
                            self.tik_instance.scalar_min(cores_with_extra_onesided, cur_core_64, cores_with_extra_onesided)
                            cores_with_normal_onesided = cur_core_64 - cores_with_extra_onesided

                            absolute_cur_move = (cores_with_extra_onesided * (onesided_move_per_core + 1)) + (cores_with_normal_onesided * onesided_move_per_core)
                            
                            with self.tik_instance.for_range(0, onesided_move_per_core + additional_onesided_move, thread_num = 2) as cur_move:
                                output_data_ub = self.tik_instance.Tensor(dtype="float32", shape=(1, 1, (self.frame_count + 4), 2), name="output_data_ub", scope=tik.scope_ubuf)
                                cur_batch = (absolute_cur_move + cur_move) // (self.n_fft // 2 + 1)
                                cur_n_fft = (absolute_cur_move + cur_move) % (self.n_fft // 2 + 1)
                                
                                
                                self.tik_instance.data_move_pad(output_data_ub, self.output_tmp_data[(cur_batch * self.n_fft * self.frame_count * 2) + cur_n_fft * self.frame_count * 2], 1, self.frame_count * 8, 0, 0)
                                self.tik_instance.data_move_pad(self.output_data[(cur_batch * (self.n_fft // 2+1) * self.frame_count * 2) + cur_n_fft * self.frame_count * 2], output_data_ub, 1, self.frame_count * 8, 0, 0)
                    else:
                        with self.tik_instance.if_scope(frame_i == 0):
                            with self.tik_instance.for_range(0, overall_onesided_moves, thread_num = 2) as cur_move:
                                output_data_ub = self.tik_instance.Tensor(dtype="float32", shape=(1, 1, (self.frame_count + 4), 2), name="output_data_ub", scope=tik.scope_ubuf)
                                cur_batch = cur_move // (self.n_fft // 2 + 1)
                                cur_n_fft = cur_move % (self.n_fft // 2 + 1)

                                self._data_move_pad_310p(output_data_ub, self.output_tmp_data[(cur_batch * self.n_fft * self.frame_count * 2) + cur_n_fft * self.frame_count * 2], self.frame_count * 2)
                                self._data_move_pad_310p(self.output_data_310p[(cur_batch * (self.n_fft // 2+1) * self.frame_count * 2) + cur_n_fft * self.frame_count * 2], output_data_ub, self.frame_count * 2, True)
                                                                                    
                    if self.overall_frame_count != 1:
                        self.tik_instance.block_barrier(self.sync_workspace)
                            
        else:
            self.transpose_16_per_core = self.transpose_16_overall // self.cores_num_910B
            self.transpose_16_left_over = self.transpose_16_overall % self.cores_num_910B

            if not self.ascend910b:
                self.transpose_16_per_core = self.transpose_16_overall
                self.transpose_16_left_over = 0
                
            if self.onesided:
                overall_onesided_moves = self.batches * (self.n_fft // 2 + 1)
                onesided_move_per_core = overall_onesided_moves // self.cores_num_910B
                onesided_move_left_over = overall_onesided_moves % self.cores_num_910B

            with self.tik_instance.for_range(0, self.cores_num_910B, block_num = self.cores_num_910B) as cur_core:
                additional_frame = self.tik_instance.Scalar(dtype="uint8", init_value=cur_core < self.left_over_frames)
                additional_batch = self.tik_instance.Scalar(dtype="uint8", init_value=cur_core < self.left_over_batches)
                
                cores_with_extra_batches = self.tik_instance.Scalar(dtype="int64", init_value=self.left_over_batches)
                cur_core = self.tik_instance.Scalar(dtype="int64", init_value=cur_core)
                self.tik_instance.scalar_min(cores_with_extra_batches, cur_core, cores_with_extra_batches)
                cores_with_normal_batches = cur_core - cores_with_extra_batches
                
                if self.center:
                    self._count_pads_parallel(additional_batch, cores_with_extra_batches, cores_with_normal_batches)

                self.tik_instance.block_barrier(self.sync_workspace)

                with self.tik_instance.new_stmt_scope():
                    data_real_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft,), name="data_real_ub", scope=tik.scope_ubuf)
                    data_imag_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft,), name="data_imag_ub", scope=tik.scope_ubuf)
                    window_data_ub = None
                    coeff_data_real_ub = None
                    coeff_data_imag_ub = None
                    tmp_ub = None
                    if self.n_fft < 4096:
                        window_data_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft,), name="window_data_ub", scope=tik.scope_ubuf)
                        coeff_data_real_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_coeff,), name="coeff_data_real_ub", scope=tik.scope_ubuf)
                        coeff_data_imag_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_coeff,), name="coeff_data_imag_ub", scope=tik.scope_ubuf)
                        tmp_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_fft * 2,), name="tmp_ub", scope=tik.scope_ubuf)

                        self.tik_instance.data_move(coeff_data_real_ub, self.coeff_data_real, 0, 1, self.n_coeff * 4 // 32, 0, 0)
                        self.tik_instance.data_move(coeff_data_imag_ub, self.coeff_data_imag, 0, 1, self.n_coeff * 4 // 32, 0, 0)

                    if self.n_fft == 400:
                        custom_masks_ub = self.tik_instance.Tensor(dtype="uint32", shape=(50, 16), name="custom_masks_ub", scope=tik.scope_ubuf)
                        
                        counter = 0
                        for i in range(2):
                            for j in range(5):
                                for k in range(5):
                                    if self.ascend910b:
                                        self.tik_instance.data_move_pad(custom_masks_ub[counter * 16], self.custom_masks_400[i][j][k], 1, (416 // 32) * 4, 0, 0, 0, 0)
                                    else:
                                        self._data_move_pad_310p(custom_masks_ub[counter * 16 :], self.custom_masks_400[i][j][k], 416 // 32)
                                    counter += 1

                    with self.tik_instance.for_range(0, self.frames_per_core + additional_frame) as cur_i:
                        cores_with_extra_frames = self.tik_instance.Scalar(dtype="int64", init_value=self.left_over_frames)
                        self.tik_instance.scalar_min(cores_with_extra_frames, cur_core, cores_with_extra_frames)
                        cores_with_normal_frames = cur_core - cores_with_extra_frames

                        frame_i = (cores_with_extra_frames * (self.frames_per_core + 1)) + (cores_with_normal_frames * self.frames_per_core) + cur_i

                        self._fft_frame(frame_i, data_real_ub, data_imag_ub, window_data_ub, coeff_data_real_ub, coeff_data_imag_ub, tmp_ub, custom_masks_ub)
                    
                self.tik_instance.block_barrier(self.sync_workspace)
                
                if self.ascend910b:
                    with self.tik_instance.if_scope(cur_core < self.transpose_16_overall):
                        additional_transpose = self.tik_instance.Scalar(dtype="uint8", init_value=cur_core < self.transpose_16_left_over)
                        self.transpose_data_input_ub = self.tik_instance.Tensor(dtype="float32", shape=(16, self.second_elements), name="transpose_data_input_ub", scope=tik.scope_ubuf)
                        self.transpose_data_output_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.second_elements, 16), name="transpose_data_output_ub", scope=tik.scope_ubuf)
                        self._parallel_transpose(cur_core, additional_transpose)
                else:
                    with self.tik_instance.if_scope(cur_core == 0):
                        self.transpose_data_input_ub = self.tik_instance.Tensor(dtype="float32", shape=(16, self.second_elements), name="transpose_data_input_ub", scope=tik.scope_ubuf)
                        self.transpose_data_output_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.second_elements, 16), name="transpose_data_output_ub", scope=tik.scope_ubuf)
                        self._parallel_transpose(cur_core, 0)
                    
                self.tik_instance.block_barrier(self.sync_workspace)

                if self.onesided:
                    if self.ascend910b:
                        with self.tik_instance.if_scope(cur_core < overall_onesided_moves):                    
                            additional_onesided_move = self.tik_instance.Scalar(dtype="uint8", init_value=cur_core < onesided_move_left_over)
                            
                            cores_with_extra_onesided = self.tik_instance.Scalar(dtype="int64", init_value=onesided_move_left_over)
                            cur_core_64 = self.tik_instance.Scalar(dtype="int64", init_value=cur_core)
                            self.tik_instance.scalar_min(cores_with_extra_onesided, cur_core_64, cores_with_extra_onesided)
                            cores_with_normal_onesided = cur_core_64 - cores_with_extra_onesided

                            absolute_cur_move = (cores_with_extra_onesided * (onesided_move_per_core + 1)) + (cores_with_normal_onesided * onesided_move_per_core)
                            
                            with self.tik_instance.for_range(0, onesided_move_per_core + additional_onesided_move, thread_num = 2) as cur_move:
                                output_data_ub = self.tik_instance.Tensor(dtype="float32", shape=(1, 1, (self.frame_count + 4), 2), name="output_data_ub", scope=tik.scope_ubuf)
                                cur_batch = (absolute_cur_move + cur_move) // (self.n_fft // 2 + 1)
                                cur_n_fft = (absolute_cur_move + cur_move) % (self.n_fft // 2 + 1)
                                
                                self.tik_instance.data_move_pad(output_data_ub, self.output_tmp_data[(cur_batch * self.n_fft * self.frame_count * 2) + cur_n_fft * self.frame_count * 2], 1, self.frame_count * 8, 0, 0)
                                self.tik_instance.data_move_pad(self.output_data[(cur_batch * (self.n_fft // 2+1) * self.frame_count * 2) + cur_n_fft * self.frame_count * 2], output_data_ub, 1, self.frame_count * 8, 0, 0)
                    
                    else:
                        with self.tik_instance.if_scope(cur_core == 0):
                            with self.tik_instance.for_range(0, overall_onesided_moves, thread_num = 2) as cur_move:
                                output_data_ub = self.tik_instance.Tensor(dtype="float32", shape=(1, 1, (self.frame_count + 4), 2), name="output_data_ub", scope=tik.scope_ubuf)
                                cur_batch = cur_move // (self.n_fft // 2 + 1)
                                cur_n_fft = cur_move % (self.n_fft // 2 + 1)

                                self._data_move_pad_310p(output_data_ub, self.output_tmp_data[(cur_batch * self.n_fft * self.frame_count * 2) + cur_n_fft * self.frame_count * 2], self.frame_count * 2)
                                self._data_move_pad_310p(self.output_data_310p[(cur_batch * (self.n_fft // 2+1) * self.frame_count * 2) + cur_n_fft * self.frame_count * 2], output_data_ub, self.frame_count * 2, True)
                    
                    self.tik_instance.block_barrier(self.sync_workspace)

        if self.ascend910b:
            if not self.onesided:     
                self.output_data = self.output_tmp_data
        else:
            if self.onesided:
                self.output_data = self.output_data_310p[:self.batches,:,:,:]
            else:
                self.output_data = self.output_tmp_data[:self.batches,:,:,:]
        
            
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_data, self.window_data], outputs=[self.output_data], config={"enable_const_fold": True})
        return self.tik_instance


def stft(x, window, y, n_fft, hop_length=128, win_length=0, center=False, pad_mode="reflect", normalized=False, onesided=True, return_complex=True, kernel_name="stft"):
    stft_obj = STFT(x, window, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided, y, kernel_name)
    stft_obj.stft_compute()