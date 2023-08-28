from te import tik
from tbe.common.platform import set_current_compile_soc_info, get_soc_spec
from numpy import pi, cos, sin, log2
from math import sqrt, ceil
from functools import reduce
import warnings

class FFT1D:
    def __init__(self, input, output, n, norm, mode, forward, kernel_name):
        """
        The FFT1D computes the Fourier transform of the input.
        @param input Indicates input tensor information
        @param output Indicates output tensor information
        @param kernel_name Indicates the kernel name of the function in the generated binary code
        """
        self.input = input
        self.output = output
        self.norm = norm
        self.mode = mode.lower()
        self.kernel_name = kernel_name
        self.input_shape = input.get("shape")
        self.input_dtype = input.get("dtype")
        self.output_shape = output.get("shape")
        self.output_dtype = output.get("dtype")

        self.forward = forward
        self.last_dim = self.input_shape[-2] if self.mode != "r2c" else self.input_shape[-1]
        self.n = n if n != 0 else self.last_dim
        if n == 0 and self.mode == "c2r":
            self.n = 2 * (self.last_dim - 1)

        self.available_factors = list(range(1, 17))
        self.factors = list()
        self.successful_factorize = self._count_factors()

        self._input_check()
        self._output_check()
        self._param_check()
        set_current_compile_soc_info("Ascend910B1", "AiCore")
        
        self.tik_instance = tik.Tik(disable_debug=False)

        self.input_data = self.tik_instance.Tensor(dtype="float32", shape=self.input_shape, name="input_data", scope=tik.scope_gm)
        self.output_data = self.tik_instance.Tensor(dtype="float32", shape=self.output_shape, name="output_data", scope=tik.scope_gm)

        self.twiddle_list = list()
        self.dft_matrices_list = list()

        self._count_twiddle()
        self._count_dft_matrices()

        self.cores_num_910B = get_soc_spec("CORE_NUM")
        self.overall_ffts = self.output_shape[-3] * self.output_shape[-4] if self.mode != "c2r" else self.output_shape[-2] * self.output_shape[-3]
        self.ffts_per_core = self.overall_ffts // self.cores_num_910B
        self.left_over_ffts = self.overall_ffts % self.cores_num_910B
        
        self.sync_workspace = self.tik_instance.Tensor(dtype="int64", shape=(min(self.cores_num_910B, self.overall_ffts) * 4,), name="sync_workspace", 
                                                       scope=tik.scope_gm, is_workspace=True, is_atomic_add=True)

        self.reserve_elements = 16 ** (len(self.factors) + 2)
        #self.reserve_elements = 16 * ceil((n // self.factors[-1]) / 8) * 8
        self.cube_gm_data = self.tik_instance.Tensor(dtype="float32", shape = (min(self.overall_ffts, self.cores_num_910B), 2, self.reserve_elements ), 
                                                     name="cube_gm_data", scope=tik.scope_gm, is_workspace=True)
        
        self.tmp_transpose_data = None
        self.tmp_transpose_data = self.tik_instance.Tensor(dtype="float32", shape = (min(self.overall_ffts, self.cores_num_910B), 2, self.reserve_elements), 
                                                            name="tmp_transpose_data", scope=tik.scope_gm, is_workspace=True)

    def _input_check(self):
        if self.mode != "r2c":
            if len(self.input_shape) != 4 or self.input_shape[-1] != 2:
                raise ValueError("Input shape doesn't match")
        else:
            if len(self.input_shape) != 3:
                raise ValueError("Input shape doesn't match")
        if self.input_dtype != "float32":
            raise ValueError("Input data type doesn't match")
        if not self.successful_factorize:
            raise ValueError("The length of the signal must be factorizable by [2, 16]")
        
    def _param_check(self):
        norms = ("forward", "backward", "ortho")
        modes = ("c2c", "r2c", "c2r")
        if type(self.n) is not int:
            raise ValueError("n parameter is not int type")
        if type(self.norm) is not str:
            raise ValueError("norm parameter is not str type")
        if not self.norm.lower() in norms:
            raise ValueError("norm supports the following normalization: \"forward\", \"backward\" and \"ortho\"")
        if type(self.mode) is not str:
            raise ValueError("mode parameter is not str type")
        if not self.mode.lower() in modes:
            raise ValueError("mode supports the following modes: \"c2c\", \"r2c\" and \"c2r\"")
        if self.n < 0:
            raise ValueError("n parameter must not be negative")
        if (self.mode == "r2c" and self.forward == False) or (self.mode == "c2r" and self.forward == True):
            warnings.warn("Passed forward parameter is not supported for the chosen mode. Switching forward to supported version...")
            self.forward = not self.forward

    def _output_check(self):
        symmetry_change = self.n // 2 + 1 if self.mode == "r2c" else self.n

        if self.mode != "c2r":
            input_shape = self.input_shape[-4:-2] if self.mode != "r2c" else self.input_shape[-3:-1]
            if len(self.output_shape) != 4 or self.output_shape[-1] != 2 or self.output_shape[-4:-2] != input_shape or self.output_shape[-2] != symmetry_change:
                raise ValueError("Output shape doesn't match")
        else:
            if len(self.output_shape) != 3 or self.output_shape[-3:-1] != self.input_shape[-4:-2] or self.output_shape[-1] != symmetry_change:
                raise ValueError("Output shape doesn't match")
        if self.output_dtype != "float32":
            raise ValueError("Output data type doesn't match")
        
    def _divide_by_factor(self, tmp_n, cur_factor_index):
        cur_factor = self.available_factors[cur_factor_index]

        while tmp_n % cur_factor == 0:
            tmp_n //= cur_factor
            self.factors.append(cur_factor)

        return tmp_n

    def _count_factors(self):
        tmp_n = self.n
        cur_factor_index = -1
        factor_len = len(self.available_factors)

        while abs(cur_factor_index) != factor_len:
            tmp_n = self._divide_by_factor(tmp_n, cur_factor_index)
            cur_factor_index -= 1
        
        if len(self.factors) == 1:
            self.factors.append(1)

        return tmp_n == 1
    
    def _dft_matrix(self, n1, n2, n):
        param = [(-1) ** int(self.forward) * 2 * pi * i * j / n for j in range(0, n1) for i in range(0, n2)]
        real = cos(param).tolist()
        imag = sin(param).tolist()
        return real, imag
    
    def _count_twiddle(self):
        mul_func = lambda x, y: x * y

        for i in range(1, len(self.factors)):
            tw_dict = {}
            n1 = self.factors[i]
            n2 = reduce(mul_func, self.factors[:i])

            tw_real_init_val, tw_imag_init_val = self._dft_matrix(n1, n2, n1 * n2)
            tw_dict["real"] = self.tik_instance.Tensor(dtype="float32", shape=(n1, n2), name=f"tw_real_{i}", 
                                                       scope=tik.scope_gm, init_value=tw_real_init_val)
            tw_dict["imag"] = self.tik_instance.Tensor(dtype="float32", shape=(n1, n2), name=f"tw_imag_{i}", 
                                                       scope=tik.scope_gm, init_value=tw_imag_init_val)
            
            self.twiddle_list.append(tw_dict)

    #m_old == k
    def _mk_to_k1mk0_gm(self, matrix, k1, m_new, k0, m_old):
        new_matrix = list()
        left_over = m_old % k0

        for j in range(k1):
            for i in range(m_old):
                start_offset = i * m_old + j * k0
                if j == (k1 - 1) and left_over:
                    end_offset = start_offset + min(left_over, k0)
                else:
                    end_offset = start_offset + k0

                to_move = matrix[start_offset : end_offset]

                new_matrix += to_move
                garbage_elements = k0 - len(to_move)
                if garbage_elements:
                    garbage_list = [0 for _ in range(garbage_elements)]
                    new_matrix += garbage_list

            for k in range(m_new - m_old):
                new_matrix += [0 for _ in range(k0)]
        return new_matrix

    def _count_dft_matrices(self):
        for i in range(len(self.factors)):
            factor = self.factors[i]
            f_dict = {}
            
            f_real_init_val, f_imag_init_val = self._dft_matrix(factor, factor, factor)
            
            k0 = 8
            k1 = ceil(factor / k0)
            m_new = ceil(factor / 16) * 16

            f_real_k1mk0 = self._mk_to_k1mk0_gm(f_real_init_val, k1, m_new, k0, factor)
            f_imag_k1mk0 = self._mk_to_k1mk0_gm(f_imag_init_val, k1, m_new, k0, factor)

            f_dict["real"] = self.tik_instance.Tensor(dtype="float32", shape=(k1, m_new, k0), name=f"f_real_{i}", scope=tik.scope_gm, init_value=f_real_k1mk0)
            f_dict["imag"] = self.tik_instance.Tensor(dtype="float32", shape=(k1, m_new, k0), name=f"f_imag_{i}", scope=tik.scope_gm, init_value=f_imag_k1mk0)

            self.dft_matrices_list.append(f_dict)
    
    def _move_complex(self, tmp_ub, data_real_ub, data_imag_ub):
        mask = self.n_to_move * 2

        self.tik_instance.vreduce(mask, data_real_ub, tmp_ub, 1, 1, 1, 8, 0, 0, None, "counter")
        self.tik_instance.vreduce(mask, data_imag_ub, tmp_ub, 2, 1, 1, 8, 0, 0, None, "counter")
    
    def _pad_data(self, data_real_ub, data_imag_ub):
        if self.pad_num > 0:
            self.tik_instance.vec_dup(self.pad_mask, data_real_ub[self.start_pad_index:], 0, self.pad_repeat_times, 8)
            if self.mode != "r2c":
                self.tik_instance.vec_dup(self.pad_mask, data_imag_ub[self.start_pad_index:], 0, self.pad_repeat_times, 8)

            if self.pad_mask_left_over:
                self.tik_instance.vec_dup(self.pad_mask_left_over, data_real_ub[self.start_pad_index + self.pad_elems_multiplied:], 0, 1, 8)
                if self.mode != "r2c":
                    self.tik_instance.vec_dup(self.pad_mask_left_over, data_imag_ub[self.start_pad_index + self.pad_elems_multiplied:], 0, 1, 8)

    def _mirror_c2r_case(self, data_real_ub, data_imag_ub):
        is_even = (self.n % 2) == 0
        if self.n >= ((self.last_dim - 1) * 2 + 1):
            is_even = 0

        with self.tik_instance.new_stmt_scope():
            tmp_scalar = self.tik_instance.Scalar(dtype="float32", name="tmp_scalar")

            with self.tik_instance.for_range(0, self.mirror_num) as i:
                data_real_ub[self.n - self.mirror_num + i].set_as(data_real_ub[self.n_to_move - 1 - is_even - i])
                tmp_scalar.set_as(data_imag_ub[self.n_to_move - 1 - is_even - i])
                data_imag_ub[self.n - self.mirror_num + i].set_as(-tmp_scalar)

    def _normalize(self, data_real_ub, data_imag_ub):
        self.tik_instance.vmuls(self.vec_mask, data_real_ub, data_real_ub, self.norm_num, self.vec_repeat_times, 1, 1, 8, 8)
        if self.mode != "r2c":
            self.tik_instance.vmuls(self.vec_mask, data_imag_ub, data_imag_ub, self.norm_num, self.vec_repeat_times, 1, 1, 8, 8)
        if self.vec_mask_left_over > 0: 
            self.tik_instance.vmuls(self.vec_mask_left_over, data_real_ub[self.elems_multiplied:], data_real_ub[self.elems_multiplied:], 
                                    self.norm_num, 1, 1, 1, 8, 8)
            if self.mode != "r2c":
                self.tik_instance.vmuls(self.vec_mask_left_over, data_imag_ub[self.elems_multiplied:], data_imag_ub[self.elems_multiplied:], 
                                        self.norm_num, 1, 1, 1, 8, 8)

    def _count_matmul_shapes(self, m_ex, k_ex, n_ex):
        n0 = 16
        m0 = 16
        k0 = 8

        m_new = int(ceil(m_ex / m0) * m0)
        k1 = int(ceil(k_ex / k0))
        n1 = int(ceil(n_ex / n0))
        n_new = n1 * n0

        return n1, n0, k1, k0, m_new, n_new

    def _transpose(self, input_ub, output_ub, elements_to_transpose, row_element_ub):
        repeat_vec_trans_times = ceil(elements_to_transpose / 8)
        
        dst_list = [output_ub[8 * i] for i in range(16)]
        src_list = [input_ub[row_element_ub * i] for i in range(16)]
        
        self.tik_instance.vec_trans_scatter(False, False, dst_list, src_list, repeat_vec_trans_times, 16, 1)
    
    def _tranpose_gm_to_k1nk0(self, tensor, shape, tmp_tensor):
        a = int(shape[0])
        b = int(shape[1])
        k0 = 8

        transpose_iterations = ceil(a / 16)
        transpose_left_over = a % 16
        transpose_second_elements = min(ceil(b / 16) * 16, 1024)
        
        transpose_blocks = ceil(b / 1024)
        elements_left_over = b % 1024

        left_over_transpose_block = bool(elements_left_over)
        nburst = 16

        with self.tik_instance.new_stmt_scope():
            tmp_transpose_input_ub = self.tik_instance.Tensor(dtype="float32", shape=(16, transpose_second_elements), 
                                                        name="tmp_transpose_input_ub", scope=tik.scope_ubuf)
            tmp_transpose_output_ub = self.tik_instance.Tensor(dtype="float32", shape=(transpose_second_elements, 16), 
                                                               name="tmp_transpose_output_ub", scope=tik.scope_ubuf)

            for cur_iter in range(transpose_iterations):
                elements_to_move = 1024
                dst_stride = (transpose_second_elements - elements_to_move) >= 8
                
                if cur_iter == (transpose_iterations - 1) and transpose_left_over:
                    nburst = transpose_left_over

                for cur_block in range(transpose_blocks):
                    if cur_block == (transpose_blocks - 1) and left_over_transpose_block:
                        elements_to_move = elements_left_over
                        dst_stride = (transpose_second_elements - elements_to_move) * 4 // 32
                    
                    src_stride_elements = b - elements_to_move
                    if src_stride_elements % 8 or elements_to_move % 8:
                        self.tik_instance.data_move_pad(tmp_transpose_input_ub, tensor[cur_iter * 16 * b + cur_block * 1024:], nburst, elements_to_move * 4, dst_stride, src_stride_elements * 4)
                    else:
                        burst = int(ceil((elements_to_move * 4) / 32))
                        self.tik_instance.data_move(tmp_transpose_input_ub, tensor[cur_iter * 16 * b + cur_block * 1024:], 0, nburst, burst, src_stride_elements * 4 // 32, dst_stride)
                    
                    self._transpose(tmp_transpose_input_ub, tmp_transpose_output_ub, elements_to_move, transpose_second_elements)

                    if transpose_iterations == 1 and transpose_blocks == 1:
                        self.tik_instance.data_move(tensor, tmp_transpose_output_ub, 0, b, 1, 1, 0)
                        if a > 8:
                            self.tik_instance.data_move(tensor[b * k0:], tmp_transpose_output_ub[8:], 0, b, 1, 1, 0)
                    else:
                        self.tik_instance.data_move(tmp_tensor[cur_iter * 16 * b + cur_block * 8 * 1024:], tmp_transpose_output_ub, 0, elements_to_move, 1, 1, 0)
                        if a > 8 or cur_iter != (transpose_iterations - 1):
                            self.tik_instance.data_move(tmp_tensor[cur_iter * 16 * b + 8 * b + cur_block * 8 * 1024:], tmp_transpose_output_ub[8:], 0, elements_to_move, 1, 1, 0)
        
        return transpose_iterations == 1 and transpose_blocks == 1

    #input must be reshaped GM and output CBUF
    def _mk_to_k1mk0(self, k1mk0_input, k1mk0_output, k1, m, k0):
        self.tik_instance.data_move(k1mk0_output, k1mk0_input, 0, 1, k1 * m * k0, 0, 0)

    #input must be GM and output CBUF
    def _kn_to_k1nk0(self, kn_input, transpose_gm, k1nk0_input, k, n_ex, k1, n_new, k0):
        if n_ex != 1:
            orig_gm_used = self._tranpose_gm_to_k1nk0(kn_input, (k, n_ex), transpose_gm)
            #self._tranpose_gm_inplace(kn_input, (k, n_ex))
            burst = n_ex * k0 * 4 // 32
            for i in range(k1):
                if orig_gm_used:
                    self.tik_instance.data_move(k1nk0_input[i * n_new * k0], kn_input[i * n_ex * k0], 0, 1, burst, 0, 0)
                else:
                    self.tik_instance.data_move(k1nk0_input[i * n_new * k0], transpose_gm[i * n_ex * k0], 0, 1, burst, 0, 0)
        else:
            for i in range(k1):
                self.tik_instance.data_move(k1nk0_input[i * n_new * k0], kn_input[i * n_ex * k0], 0, 1, 1, 0, 0)

    def _matmul(self, gm_cube, a_k1mk0, b_k1nk0, res_n1mn0, res_ub, m_ex, k_ex, n_ex):
        n1, n0, _, _, _, _ = self._count_matmul_shapes(m_ex, k_ex, n_ex)
        self.tik_instance.matmul(res_n1mn0, a_k1mk0, b_k1nk0, m_ex, k_ex, n_ex)
        
        self.tik_instance.fixpipe(gm_cube, res_n1mn0, n1, m_ex * n0 * 4 // 32, 0, 0)
        
        if n_ex % 8 == 0 and n_ex <= n0:
            self.tik_instance.data_move(res_ub, gm_cube, 0, m_ex, n_ex * 4 // 32, (16 - n_ex) * 4 // 32, 0)
        else:
            n_ex_ub = int(ceil(n_ex / 8) * 8)

            with self.tik_instance.new_stmt_scope():
                tmp_ub = self.tik_instance.Tensor(dtype="float32", shape=(m_ex, n_ex_ub), name="tmp_ub", scope=tik.scope_ubuf)
                if n_ex < n0:
                    self.tik_instance.data_move(tmp_ub, gm_cube, 0, m_ex, n_ex_ub * 4 // 32, (16 - n_ex_ub) * 4 // 32, 0)
                    self.tik_instance.data_move_pad(gm_cube, tmp_ub, m_ex, n_ex * 4, 0, 0)
                else:
                    if n_ex % 16 > 8 or (n_ex % 16 == 0):
                        for i in range(m_ex):
                            self.tik_instance.data_move(tmp_ub[i * n_ex_ub:], gm_cube[i * n0:], 0, n1, n0 * 4 // 32, n0 * (m_ex - 1) * 4 // 32, 0)
                    else:
                        for i in range(m_ex):
                            self.tik_instance.data_move(tmp_ub[i * n_ex_ub:], gm_cube[i * n0:], 0, (n1 - 1), n0 * 4 // 32, n0 * (m_ex - 1) * 4 // 32, 0)
                            self.tik_instance.data_move(tmp_ub[i * n_ex_ub + (n1 - 1) * n0:], gm_cube[i * n0 + (m_ex * (n1 - 1) * n0)], 0, 1, 1, 0, 0)
            
                    self.tik_instance.data_move_pad(gm_cube, tmp_ub, m_ex, n_ex * 4, 0, 0)

            if m_ex * n_ex % 8 == 0:
                self.tik_instance.data_move(res_ub, gm_cube, 0, 1, m_ex * n_ex * 4 // 32, 0, 0)
            else:
                to_move = (m_ex * n_ex // 8) * 8
                pad_move = (m_ex * n_ex) % 8

                if to_move > 0:
                    self.tik_instance.data_move(res_ub, gm_cube, 0, 1, to_move * 4 // 32, 0, 0)
                self.tik_instance.data_move_pad(res_ub[to_move:], gm_cube[to_move:], 1, pad_move * 4, 0, 0)

    def _dft_cube(self, gm_cube, transpose_gm, f_k1mk0, data_k1nk0, m_ex, k_ex, n_ex, data_real_ub, data_imag_ub, first=False):
        f_real_k1mk0 = f_k1mk0[0]
        f_imag_k1mk0 = f_k1mk0[1]

        data_real_k1nk0 = data_k1nk0[0]
        data_imag_k1nk0 = data_k1nk0[1]
        
        size = m_ex * n_ex
        mask = min(64, size)
        vec_repeat_times = max(1, size // 64)
        mask_leftover = size % 64 if size > 64 else 0
        elems_multiplyed = mask * vec_repeat_times

        with self.tik_instance.new_stmt_scope():
            n1, n0, _, _, m_new, _ = self._count_matmul_shapes(m_ex, k_ex, n_ex)
            
            res_n1mn0 = self.tik_instance.Tensor(dtype="float32", shape=(n1, m_new, n0), name="res_n1mn0", scope=tik.scope_cbuf_out)

            # self._mk_to_k1mk0(f_real, f_real_k1mk0, k1, m_new, k0)
            # self._mk_to_k1mk0(f_imag, f_imag_k1mk0, k1, m_new, k0)

            # self._kn_to_k1nk0(gm_cube[:self.reserve_elements], transpose_gm, data_real_k1nk0, k_ex, n_ex, k1, n_new, k0)
            # if not first or self.mode != "r2c":
            #     self._kn_to_k1nk0(gm_cube[self.reserve_elements:], transpose_gm, data_imag_k1nk0, k_ex, n_ex, k1, n_new, k0)

            if not first or self.mode != "r2c":
                tmp_ub_1 = self.tik_instance.Tensor(dtype="float32", shape=(ceil(size / 8) * 8,), name="tmp_ub_1", scope=tik.scope_ubuf)
                tmp_ub_2 = self.tik_instance.Tensor(dtype="float32", shape=(ceil(size / 8) * 8,), name="tmp_ub_2", scope=tik.scope_ubuf)
                
                self._matmul(gm_cube[:self.reserve_elements], f_real_k1mk0, data_real_k1nk0, res_n1mn0, tmp_ub_1, m_ex, k_ex, n_ex)
                self._matmul(gm_cube[self.reserve_elements:], f_imag_k1mk0, data_imag_k1nk0, res_n1mn0, tmp_ub_2, m_ex, k_ex, n_ex)

                self.tik_instance.vec_sub(mask, data_real_ub, tmp_ub_1, tmp_ub_2, vec_repeat_times, 8, 8, 8)
                if mask_leftover:
                    self.tik_instance.vec_sub(mask_leftover, data_real_ub[elems_multiplyed:], tmp_ub_1[elems_multiplyed:], tmp_ub_2[elems_multiplyed:], 1, 8, 8, 8)
            
                self._matmul(gm_cube[:self.reserve_elements], f_real_k1mk0, data_imag_k1nk0, res_n1mn0, tmp_ub_1, m_ex, k_ex, n_ex)
                self._matmul(gm_cube[self.reserve_elements:], f_imag_k1mk0, data_real_k1nk0, res_n1mn0, tmp_ub_2, m_ex, k_ex, n_ex)

                self.tik_instance.vec_add(mask, data_imag_ub, tmp_ub_1, tmp_ub_2, vec_repeat_times, 8, 8, 8)
                if mask_leftover:
                    self.tik_instance.vec_add(mask_leftover, data_imag_ub[elems_multiplyed:], tmp_ub_1[elems_multiplyed:], tmp_ub_2[elems_multiplyed:], 1, 8, 8, 8)
            else:
                self._matmul(gm_cube[:self.reserve_elements], f_real_k1mk0, data_real_k1nk0, res_n1mn0, data_real_ub, m_ex, k_ex, n_ex)

                self._matmul(gm_cube[self.reserve_elements:], f_real_k1mk0, data_imag_k1nk0, res_n1mn0, data_imag_ub, m_ex, k_ex, n_ex)
    
    def _transpose_ub(self, data_ub, gm_tmp, shape):
        a = int(shape[0])
        b = int(shape[1])
        output_to_gm = a % 8

        transpose_iterations = ceil(a / 16)
        transpose_left_over = a % 16
        transpose_second_elements = min(ceil(b / 16) * 16, 1024)
        
        transpose_blocks = ceil(b / 1024)
        elements_left_over = b % 1024

        nburst = 16
        
        size_ub = ceil((a * b) / 8) * 8
        data_ro_read_from = data_ub

        with self.tik_instance.new_stmt_scope():
            tmp_transpose_input_ub = self.tik_instance.Tensor(dtype="float32", shape=(16, transpose_second_elements), 
                                                        name="tmp_transpose_input_ub", scope=tik.scope_ubuf)
            tmp_transpose_output_ub = self.tik_instance.Tensor(dtype="float32", shape=(transpose_second_elements, 16), 
                                                               name="tmp_transpose_output_ub", scope=tik.scope_ubuf)
            
            if transpose_iterations != 1 or transpose_blocks != 1:
                prev_data_ub = self.tik_instance.Tensor(dtype="float32", shape=(size_ub,), name="prev_data_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(prev_data_ub, data_ub, 0, 1, size_ub * 4 // 32, 0, 0)
                data_ro_read_from = prev_data_ub

            for cur_iter in range(transpose_iterations):
                elements_to_move = 1024
                dst_elements_gap = 0
                
                if cur_iter == (transpose_iterations - 1) and transpose_left_over:
                    nburst = transpose_left_over
                
                for cur_block in range(transpose_blocks):
                    if cur_block == (transpose_blocks - 1) and elements_left_over:
                        elements_to_move = elements_left_over
                        dst_elements_gap = transpose_second_elements - elements_to_move
                    
                    src_elements_gap = b - elements_to_move

                    if src_elements_gap % 8 or elements_to_move % 8:
                        tmp_elements = a * b - cur_iter * 16 * b + cur_block * 1024
                        burst = int(ceil((tmp_elements * 4) / 32))
                        self.tik_instance.data_move(gm_tmp, data_ro_read_from[cur_iter * 16 * b + cur_block * 1024:], 0, 1, tmp_elements, 0, 0)
                        self.tik_instance.data_move_pad(tmp_transpose_input_ub, gm_tmp, nburst, elements_to_move * 4, dst_elements_gap * 4 // 32, src_elements_gap * 4)
                    else:
                        burst = int(ceil((elements_to_move * 4) / 32))
                        self.tik_instance.data_move(tmp_transpose_input_ub, data_ro_read_from[cur_iter * 16 * b + cur_block * 1024:], 0, nburst, burst, 0, dst_elements_gap * 4 // 32)

                    self._transpose(tmp_transpose_input_ub, tmp_transpose_output_ub, elements_to_move, transpose_second_elements)

                    output_src_elements_gap = 16 - nburst
                    output_dst_elements_gap = a - nburst
                    if output_to_gm:
                        self.tik_instance.data_move_pad(gm_tmp[cur_iter * 16 + (a - nburst) * cur_block * 1024], tmp_transpose_output_ub, elements_to_move, nburst * 4,
                                                        output_dst_elements_gap * 4, output_src_elements_gap * 4 // 32)
                    else:
                        self.tik_instance.data_move(data_ub[cur_iter * 16 + (a - nburst) * cur_block * 1024], tmp_transpose_output_ub, 0, elements_to_move, nburst * 4 // 32, 
                                                    output_src_elements_gap * 4 // 32, output_dst_elements_gap * 4 // 32)
        
        if output_to_gm:
            self.tik_instance.data_move(data_ub, gm_tmp, 0, 1, size_ub, 0, 0)

    def _twiddle_mul_cube(self, data_real_ub, data_imag_ub, tw_tensors, n1, n2):
        size = n1 * n2
        mask = min(64, size)
        vec_repeat_times = max(1, size // 64)
        mask_leftover = size % 64 if size > 64 else 0
        elems_multiplyed = mask * vec_repeat_times

        tw_real = tw_tensors["real"]
        tw_imag = tw_tensors["imag"]
        
        size_ub = ceil(size / 8) * 8

        with self.tik_instance.new_stmt_scope():
            tw_real_ub = self.tik_instance.Tensor(dtype="float32", shape=(size_ub,), name="tw_real_ub", scope=tik.scope_ubuf)
            tw_imag_ub = self.tik_instance.Tensor(dtype="float32", shape=(size_ub,), name="tw_imag_ub", scope=tik.scope_ubuf)

            self.tik_instance.data_move_pad(tw_real_ub, tw_real, 1, size * 4, 0, 0)
            self.tik_instance.data_move_pad(tw_imag_ub, tw_imag, 1, size * 4, 0, 0)

            prev_data_real_ub = self.tik_instance.Tensor(dtype="float32", shape=(size_ub,), name="prev_data_real_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(prev_data_real_ub, data_real_ub, 0, 1, size_ub * 4 // 32, 0, 0)

            tmp_ub_1 = self.tik_instance.Tensor(dtype="float32", shape=(size,), name="tmp_ub_1", scope=tik.scope_ubuf)
            tmp_ub_2 = self.tik_instance.Tensor(dtype="float32", shape=(size,), name="tmp_ub_2", scope=tik.scope_ubuf)

            self.tik_instance.vec_mul(mask, tmp_ub_1, tw_real_ub, data_real_ub, vec_repeat_times, 8, 8, 8)
            self.tik_instance.vec_mul(mask, tmp_ub_2, tw_imag_ub, data_imag_ub, vec_repeat_times, 8, 8, 8)

            self.tik_instance.vec_sub(mask, data_real_ub, tmp_ub_1, tmp_ub_2, vec_repeat_times, 8, 8, 8)

            if mask_leftover:
                self.tik_instance.vec_mul(mask_leftover, tmp_ub_1[elems_multiplyed:], tw_real_ub[elems_multiplyed:], 
                                        data_real_ub[elems_multiplyed:], 1, 8, 8, 8)
                self.tik_instance.vec_mul(mask_leftover, tmp_ub_2[elems_multiplyed:], tw_imag_ub[elems_multiplyed:], 
                                        data_imag_ub[elems_multiplyed:], 1, 8, 8, 8)

                self.tik_instance.vec_sub(mask_leftover, data_real_ub[elems_multiplyed:], tmp_ub_1[elems_multiplyed:], 
                                        tmp_ub_2[elems_multiplyed:], 1, 8, 8, 8)
                
            self.tik_instance.vec_mul(mask, tmp_ub_1, tw_real_ub, data_imag_ub, vec_repeat_times, 8, 8, 8)
            self.tik_instance.vec_mul(mask, tmp_ub_2, tw_imag_ub, prev_data_real_ub, vec_repeat_times, 8, 8, 8)

            self.tik_instance.vec_add(mask, data_imag_ub, tmp_ub_1, tmp_ub_2, vec_repeat_times, 8, 8, 8)

            if mask_leftover:
                self.tik_instance.vec_mul(mask_leftover, tmp_ub_1[elems_multiplyed:], tw_real_ub[elems_multiplyed:], 
                                        data_imag_ub[elems_multiplyed:], 1, 8, 8, 8)
                self.tik_instance.vec_mul(mask_leftover, tmp_ub_2[elems_multiplyed:], tw_imag_ub[elems_multiplyed:], 
                                        prev_data_real_ub[elems_multiplyed:], 1, 8, 8, 8)

                self.tik_instance.vec_add(mask_leftover, data_imag_ub[elems_multiplyed:], tmp_ub_1[elems_multiplyed:], 
                                        tmp_ub_2[elems_multiplyed:], 1, 8, 8, 8)

    def _count_fft_cube(self, gm_cube, data_real_ub, data_imag_ub, transpose_gm):
        other_radices = self.n / self.factors[0]

        if self.n % 8 == 0:
            self.tik_instance.data_move(gm_cube[:self.reserve_elements], data_real_ub, 0, 1, self.n * 4 // 32, 0, 0)
            if self.mode != "r2c":
                self.tik_instance.data_move(gm_cube[self.reserve_elements:], data_imag_ub, 0, 1, self.n * 4 // 32, 0, 0)
        else:
            self.tik_instance.data_move_pad(gm_cube[:self.reserve_elements], data_real_ub, 1, self.n * 4, 0, 0)
            if self.mode != "r2c":
                self.tik_instance.data_move_pad(gm_cube[self.reserve_elements:], data_imag_ub, 1, self.n * 4, 0, 0)
        
        m_ex = int(self.factors[0])
        k_ex = int(m_ex)
        n_ex = int(other_radices)

        dft_matrix = self.dft_matrices_list[0]
        _, _, k1, k0, n_new, m_new = self._count_matmul_shapes(n_ex, k_ex, m_ex)
        #_, _, k1, k0, m_new, n_new = self._count_matmul_shapes(m_ex, k_ex, n_ex)

        f_real_k1mk0 = self.tik_instance.Tensor(dtype="float32", shape=(k1, m_new, k0), name="f_real_k1mk0", scope=tik.scope_cbuf)
        f_imag_k1mk0 = self.tik_instance.Tensor(dtype="float32", shape=(k1, m_new, k0), name="f_imag_k1mk0", scope=tik.scope_cbuf)

        self._mk_to_k1mk0(dft_matrix["real"], f_real_k1mk0, k1, m_new, k0)
        self._mk_to_k1mk0(dft_matrix["imag"], f_imag_k1mk0, k1, m_new, k0)
        
        f_k1mk0 = (f_real_k1mk0, f_imag_k1mk0)
        
        with self.tik_instance.new_stmt_scope():
            data_real_k1nk0 = self.tik_instance.Tensor(dtype="float32", shape=(k1, n_new, k0), name="data_real_k1nk0", scope=tik.scope_cbuf)
            data_imag_k1nk0 = None
            if self.mode != "r2c" or len(self.factors) > 1:
                data_imag_k1nk0 = self.tik_instance.Tensor(dtype="float32", shape=(k1, n_new, k0), name="data_imag_k1nk0", scope=tik.scope_cbuf)

            self._kn_to_k1nk0(gm_cube[:self.reserve_elements], transpose_gm, data_real_k1nk0, k_ex, n_ex, k1, n_new, k0)
            if self.mode != "r2c":
                self._kn_to_k1nk0(gm_cube[self.reserve_elements:], transpose_gm, data_imag_k1nk0, k_ex, n_ex, k1, n_new, k0)

            data_k1nk0 = (data_real_k1nk0, data_imag_k1nk0)

            self._dft_cube(gm_cube, transpose_gm, data_k1nk0, f_k1mk0, n_ex, k_ex, m_ex, data_real_ub, data_imag_ub, first=True)

        # if self.factors[-1] != 1:
        #     self._transpose_ub(data_real_ub, gm_cube[:self.reserve_elements], (k_ex, n_ex))
        #     self._transpose_ub(data_imag_ub, gm_cube[self.reserve_elements:], (k_ex, n_ex))

        if len(self.factors) > 2:
            mul_func = lambda x, y: x * y
            for cur_radix_index in range(1, len(self.factors) - 1):
                cur_radix = self.factors[cur_radix_index]
                next_radices = reduce(mul_func, self.factors[cur_radix_index + 1:])
                prev_radices = reduce(mul_func, self.factors[:cur_radix_index])

                m_ex = cur_radix
                k_ex = int(cur_radix)
                n_ex = int(prev_radices)

                _, _, k1, k0, m_new, n_new = self._count_matmul_shapes(m_ex, k_ex, n_ex)

                if cur_radix != self.factors[cur_radix_index - 1]:
                    dft_matrix = self.dft_matrices_list[cur_radix_index]
                    self._mk_to_k1mk0(dft_matrix["real"], f_real_k1mk0, k1, m_new, k0)
                    self._mk_to_k1mk0(dft_matrix["imag"], f_imag_k1mk0, k1, m_new, k0)

                with self.tik_instance.new_stmt_scope():
                    tmp_data_real_ub = None
                    tmp_data_imag_ub = None

                    tmp_data_ub_size = ceil(k_ex * n_ex / 8) * 8

                    tmp_data_real_ub = self.tik_instance.Tensor(dtype="float32", shape=(next_radices, tmp_data_ub_size), name="tmp_data_real_ub", scope=tik.scope_ubuf)
                    tmp_data_imag_ub = self.tik_instance.Tensor(dtype="float32", shape=(next_radices, tmp_data_ub_size), name="tmp_data_imag_ub", scope=tik.scope_ubuf)

                    if n_ex % 8:
                        with self.tik_instance.new_stmt_scope():
                            tmp_ub_1 = self.tik_instance.Tensor(dtype="float32", shape=(k_ex, ceil(n_ex / 8) * 8), name="tmp_ub_1", scope=tik.scope_ubuf)
                            tmp_ub_2 = self.tik_instance.Tensor(dtype="float32", shape=(k_ex, ceil(n_ex / 8) * 8), name="tmp_ub_1", scope=tik.scope_ubuf)
                        
                            for cur_iter in range(next_radices):
                                self.tik_instance.data_move_pad(gm_cube[:self.reserve_elements], data_real_ub, 1, self.n * 4, 0, 0)
                                self.tik_instance.data_move_pad(gm_cube[self.reserve_elements:], data_imag_ub, 1, self.n * 4, 0, 0)

                                self.tik_instance.data_move_pad(tmp_ub_1, gm_cube[cur_iter * n_ex:self.reserve_elements], k_ex, n_ex * 4, 0, (next_radices - 1) * prev_radices * 4)
                                self.tik_instance.data_move_pad(tmp_ub_2, gm_cube[self.reserve_elements + cur_iter * n_ex:], k_ex, n_ex * 4, 0, (next_radices - 1) * prev_radices * 4)
                                self.tik_instance.data_move_pad(gm_cube[:self.reserve_elements], tmp_ub_1, k_ex, n_ex * 4, 0, 0)
                                self.tik_instance.data_move_pad(gm_cube[self.reserve_elements:], tmp_ub_2, k_ex, n_ex * 4, 0, 0)
                                if k_ex * n_ex % 8:
                                    self.tik_instance.data_move_pad(tmp_data_real_ub[cur_iter * tmp_data_ub_size:], gm_cube[:self.reserve_elements], 1, k_ex * n_ex * 4, 0, 0)
                                    self.tik_instance.data_move_pad(tmp_data_imag_ub[cur_iter * tmp_data_ub_size:], gm_cube[self.reserve_elements:], 1, k_ex * n_ex * 4, 0, 0)
                                else:
                                    self.tik_instance.data_move(tmp_data_real_ub[cur_iter * tmp_data_ub_size:], gm_cube[:self.reserve_elements], 0, 1, k_ex * n_ex, 0, 0)
                                    self.tik_instance.data_move(tmp_data_imag_ub[cur_iter * tmp_data_ub_size:], gm_cube[self.reserve_elements:], 0, 1, k_ex * n_ex, 0, 0)
                                
                                self._twiddle_mul_cube(tmp_data_real_ub[cur_iter * tmp_data_ub_size:], tmp_data_imag_ub[cur_iter * tmp_data_ub_size:], 
                                                self.twiddle_list[cur_radix_index - 1], k_ex, n_ex)

                    with self.tik_instance.new_stmt_scope():
                        data_real_k1nk0 = self.tik_instance.Tensor(dtype="float32", shape=(k1, n_new, k0), name="data_real_k1nk0", scope=tik.scope_cbuf)
                        data_imag_k1nk0 = self.tik_instance.Tensor(dtype="float32", shape=(k1, n_new, k0), name="data_imag_k1nk0", scope=tik.scope_cbuf)

                        for cur_iter in range(next_radices):
                            if n_ex % 8 == 0:
                                self.tik_instance.data_move(tmp_data_real_ub[cur_iter * k_ex * n_ex:], data_real_ub[cur_iter * n_ex:], 0, k_ex, n_ex * 4 // 32, 
                                                            (next_radices - 1) * prev_radices * 4 // 32, 0)
                                self.tik_instance.data_move(tmp_data_imag_ub[cur_iter * k_ex * n_ex:], data_imag_ub[cur_iter * n_ex:], 0, k_ex, n_ex * 4 // 32, 
                                                            (next_radices - 1) * prev_radices * 4 // 32, 0)
                                
                                self._twiddle_mul_cube(tmp_data_real_ub[cur_iter * tmp_data_ub_size:], tmp_data_imag_ub[cur_iter * tmp_data_ub_size:], 
                                                    self.twiddle_list[cur_radix_index - 1], k_ex, n_ex)

                            if k_ex * n_ex % 8 == 0:        
                                self.tik_instance.data_move(gm_cube[:self.reserve_elements], tmp_data_real_ub[cur_iter * tmp_data_ub_size:], 0, 1, k_ex * n_ex * 4 // 32, 0, 0)
                                self.tik_instance.data_move(gm_cube[self.reserve_elements:], tmp_data_imag_ub[cur_iter * tmp_data_ub_size:], 0, 1, k_ex * n_ex * 4 // 32, 0, 0)
                            else:
                                self.tik_instance.data_move_pad(gm_cube[:self.reserve_elements], tmp_data_real_ub[cur_iter * tmp_data_ub_size:], 1, k_ex * n_ex * 4, 0, 0)
                                self.tik_instance.data_move_pad(gm_cube[self.reserve_elements:], tmp_data_imag_ub[cur_iter * tmp_data_ub_size:], 1, k_ex * n_ex * 4, 0, 0)

                            self._kn_to_k1nk0(gm_cube[:self.reserve_elements], transpose_gm, data_real_k1nk0, k_ex, n_ex, k1, n_new, k0)
                            self._kn_to_k1nk0(gm_cube[self.reserve_elements:], transpose_gm, data_imag_k1nk0, k_ex, n_ex, k1, n_new, k0)

                            data_k1nk0 = (data_real_k1nk0, data_imag_k1nk0)

                            self._dft_cube(gm_cube, transpose_gm, f_k1mk0, data_k1nk0, m_ex, k_ex, n_ex, 
                                        tmp_data_real_ub[cur_iter * tmp_data_ub_size:], tmp_data_imag_ub[cur_iter * tmp_data_ub_size:])
                    
                    if k_ex * n_ex % 8 == 0:
                        self.tik_instance.data_move(data_real_ub, tmp_data_real_ub, 0, 1, self.n * 4 // 32, 0, 0)
                        self.tik_instance.data_move(data_imag_ub, tmp_data_imag_ub, 0, 1, self.n * 4 // 32, 0, 0)
                    else:
                        self.tik_instance.data_move_pad(gm_cube[:self.reserve_elements], tmp_data_real_ub, next_radices, k_ex * n_ex * 4, 0, 0)
                        self.tik_instance.data_move_pad(gm_cube[self.reserve_elements:], tmp_data_imag_ub, next_radices, k_ex * n_ex * 4, 0, 0)

                        self.tik_instance.data_move_pad(data_real_ub, gm_cube[:self.reserve_elements], 1, next_radices * k_ex * n_ex * 4, 0, 0)
                        self.tik_instance.data_move_pad(data_imag_ub, gm_cube[self.reserve_elements:], 1, next_radices * k_ex * n_ex * 4, 0, 0)

        if (self.factors[-1] != 1):
            m_ex = self.factors[-1]
            k_ex = m_ex
            n_ex = self.n // k_ex

            dft_matrix = self.dft_matrices_list[-1]

            with self.tik_instance.new_stmt_scope():
                self._twiddle_mul_cube(data_real_ub, data_imag_ub, self.twiddle_list[-1], k_ex, n_ex)
                if self.n % 8 == 0:
                    self.tik_instance.data_move(gm_cube[:self.reserve_elements], data_real_ub, 0, 1, self.n * 4 // 32, 0, 0)
                    self.tik_instance.data_move(gm_cube[self.reserve_elements:], data_imag_ub, 0, 1, self.n * 4 // 32, 0, 0)
                else:
                    self.tik_instance.data_move_pad(gm_cube[:self.reserve_elements], data_real_ub, 1, self.n * 4, 0, 0)
                    self.tik_instance.data_move_pad(gm_cube[self.reserve_elements:], data_imag_ub, 1, self.n * 4, 0, 0)

                _, _, k1, k0, m_new, n_new = self._count_matmul_shapes(m_ex, k_ex, n_ex)

                if self.factors[-1] != self.factors[-2]:
                    dft_matrix = self.dft_matrices_list[-1]
                    self._mk_to_k1mk0(dft_matrix["real"], f_real_k1mk0, k1, m_new, k0)
                    self._mk_to_k1mk0(dft_matrix["imag"], f_imag_k1mk0, k1, m_new, k0)

                with self.tik_instance.new_stmt_scope():
                    data_real_k1nk0 = self.tik_instance.Tensor(dtype="float32", shape=(k1, n_new, k0), name="data_real_k1nk0", scope=tik.scope_cbuf)
                    data_imag_k1nk0 = self.tik_instance.Tensor(dtype="float32", shape=(k1, n_new, k0), name="data_imag_k1nk0", scope=tik.scope_cbuf)
                    
                    self._kn_to_k1nk0(gm_cube[:self.reserve_elements], transpose_gm, data_real_k1nk0, k_ex, n_ex, k1, n_new, k0)
                    self._kn_to_k1nk0(gm_cube[self.reserve_elements:], transpose_gm, data_imag_k1nk0, k_ex, n_ex, k1, n_new, k0)
                    
                    data_k1nk0 = (data_real_k1nk0, data_imag_k1nk0)

                    self._dft_cube(gm_cube, transpose_gm, f_k1mk0, data_k1nk0, m_ex, k_ex, n_ex, data_real_ub, data_imag_ub)

    def _count_fft_i(self, gm_cube, fft_i, data_real_ub, data_imag_ub, transpose_gm = None):
        self._pad_data(data_real_ub, data_imag_ub)

        if self.mode != "r2c":
            with self.tik_instance.new_stmt_scope():
                tmp_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.n_to_move * 2 + 8,), name="tmp_ub", scope=tik.scope_ubuf)
                if (self.n_to_move * 2) % 8:
                    self.tik_instance.data_move_pad(tmp_ub, self.input_data[fft_i * self.last_dim * 2], 1, self.n_to_move * 2 * 4, 0, 0)
                else:
                    self.tik_instance.data_move(tmp_ub, self.input_data[fft_i * self.last_dim * 2], 0, 1, self.n_to_move * 2  * 4 // 32, 0, 0)
                self._move_complex(tmp_ub, data_real_ub, data_imag_ub)
        else:
            if self.n_to_move % 8:
                right_padding = ceil(self.n_to_move / 8) * 8 - self.n_to_move 
                self.tik_instance.data_move_pad(data_real_ub, self.input_data[fft_i * self.last_dim], 1, self.n_to_move * 4, 0, 0, right_padding, 0, 0)
            else:
                self.tik_instance.data_move(data_real_ub, self.input_data[fft_i * self.last_dim], 0, 1, self.n_to_move * 4 // 32, 0, 0)
        
        if self.to_normalize:
            self._normalize(data_real_ub, data_imag_ub)
        
        if self.mode == "c2r":
            self._mirror_c2r_case(data_real_ub, data_imag_ub)

        self._count_fft_cube(gm_cube, data_real_ub, data_imag_ub, transpose_gm)
        
    def _transpose_and_write_real_imag(self, fft_i, data_real_ub, data_imag_ub):
        transpose_elements_ub = min(max(16, self.n_out), 1024)
        transpose_elements_ub = ceil(transpose_elements_ub / 16) * 16

        transpose_blocks = ceil(self.n_out / 1024)
        elements_to_transpose = 1024
        left_over_elements = self.n_out % 1024
        elements_to_move = elements_to_transpose
        
        with self.tik_instance.new_stmt_scope():
            transpose_input_ub = self.tik_instance.Tensor(dtype="float32", shape=(16, transpose_elements_ub), name="transpose_input_ub", scope=tik.scope_ubuf)
            transpose_output_ub = self.tik_instance.Tensor(dtype="float32", shape=(transpose_elements_ub, 16), name="transpose_output_ub", scope=tik.scope_ubuf)
            
            for cur_block in range(transpose_blocks):
                if cur_block == (transpose_blocks - 1) and left_over_elements > 0:
                    elements_to_transpose = ceil(left_over_elements / 8) * 8 
                    elements_to_move = left_over_elements

                self.tik_instance.data_move(transpose_input_ub, data_real_ub[cur_block * 1024:], 0, 1, elements_to_transpose * 4 // 32, 0, 0)
                self.tik_instance.data_move(transpose_input_ub[transpose_elements_ub:], data_imag_ub[cur_block * 1024:], 0, 1, elements_to_transpose * 4 // 32, 0, 0)

                repeat_vec_trans_times = elements_to_transpose // 8
            
                dst_list = [transpose_output_ub[8 * i] for i in range(16)]
                src_list = [transpose_input_ub[transpose_elements_ub * i] for i in range(16)]
                
                self.tik_instance.vec_trans_scatter(False, False, dst_list, src_list, repeat_vec_trans_times, 16, 1)
                
                self.tik_instance.data_move_pad(self.output_data[(2 * fft_i * self.n_out) + cur_block * 2 * 1024], 
                                                transpose_output_ub, elements_to_move, 2 * 4, 0, 1)

    def _fft1d_compute_large_case(self, cur_core, gm_cube):
        additional_frame = self.tik_instance.Scalar(dtype="uint8", init_value=cur_core < self.left_over_ffts)

        with self.tik_instance.for_range(0, self.ffts_per_core + additional_frame) as fft_i:
            with self.tik_instance.new_stmt_scope():
                data_real_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.ub_size,), name="data_real_ub", scope=tik.scope_ubuf)
                data_imag_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.ub_size,), name="data_imag_ub", scope=tik.scope_ubuf)


    def fft1d_compute(self):
        self.ub_size = ceil(self.n / 8) * 8
        input_n = self.n // 2 + 1 if self.mode == "c2r" else self.n
        self.n_to_move = min(self.last_dim, input_n)
        self.non_move_num = self.n - self.n_to_move

        self.n_out = self.n
        if self.mode == "r2c":
            self.n_out = self.n // 2 + 1
        
        self.pad_num = self.non_move_num
        if self.mode == "c2r":
            is_even = (self.n % 2 == 0)
            if self.n >= ((self.last_dim - 1) * 2 + 1):
                is_even = 0
            self.mirror_num = min(self.last_dim, (self.n_to_move - 1) - is_even)
            self.pad_num -= self.mirror_num

        self.start_pad_index = (self.n_to_move // 8) * 8
        self.diff = self.n_to_move - self.start_pad_index
        self.pad_mask = min(64, (self.pad_num + self.diff))
        self.pad_repeat_times = max(1, (self.pad_num + self.diff) // 64)
        self.pad_mask_left_over = (self.pad_num + self.diff) % 64 if (self.pad_num + self.diff) > 64 else 0

        if self.pad_mask_left_over:
            self.pad_elems_multiplied = self.pad_mask * self.pad_repeat_times 

        self.to_normalize = (self.forward and self.norm == "forward") or (not self.forward and self.norm == "backward") or self.norm == "ortho"
        self.norm_num = 1 / self.n

        if self.norm == "ortho":
            self.norm_num = 1 / sqrt(self.n)

        self.vec_mask = min(64, self.n_to_move)
        self.vec_repeat_times = max(1, self.n_to_move // 64)
        self.vec_mask_left_over = self.n_to_move % 64 if self.n_to_move > 64 else 0
        
        if self.vec_mask_left_over:
            self.elems_multiplied = self.vec_mask * self.vec_repeat_times

        if self.n > 4096:
            self.max_ub_size = 16384 #192 kB devided by 3
            self.cur_ub_size = 16384
            self.vec_ub_ops = ceil(self.n / self.max_ub_size)
            self.vec_ub_ops_left_over = self.n % self.max_ub_size

        if self.overall_ffts < self.cores_num_910B:
            with self.tik_instance.for_range(0, self.overall_ffts, block_num = self.overall_ffts) as fft_i:
                gm_offset_start = fft_i * 2 * self.reserve_elements
                gm_offset_end = gm_offset_start + 2 * self.reserve_elements

                gm_cube = self.cube_gm_data[gm_offset_start:gm_offset_end]

                if True:
                #if self.n < 4096:
                    data_real_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.ub_size,), name="data_real_ub", scope=tik.scope_ubuf)
                    data_imag_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.ub_size,), name="data_imag_ub", scope=tik.scope_ubuf)
                    
                    self._count_fft_i(gm_cube, fft_i, data_real_ub, data_imag_ub, self.tmp_transpose_data[gm_offset_start:gm_offset_end])

                    if self.mode != "c2r":
                        self._transpose_and_write_real_imag(fft_i, data_real_ub, data_imag_ub)
                    else:
                        if (self.n % 8) == 0:
                            self.tik_instance.data_move(self.output_data[fft_i * self.n], data_real_ub, 0, 1, self.n * 4 // 32, 0, 0)
                        else:
                            self.tik_instance.data_move_pad(self.output_data[fft_i * self.n], data_real_ub, 1, self.n * 4, 0, 0)

                else:
                    self._fft1d_compute_large_case(fft_i, gm_cube)

                if self.overall_ffts != 1:
                    self.tik_instance.block_barrier(self.sync_workspace)
        else:
            with self.tik_instance.for_range(0, self.cores_num_910B, block_num = self.cores_num_910B) as cur_core:
                gm_offset_start = cur_core * 2 * self.reserve_elements
                gm_offset_end = gm_offset_start + 2 * self.reserve_elements

                gm_cube = self.cube_gm_data[gm_offset_start:gm_offset_end]

                if True:
                #if self.n < 4096:
                    additional_frame = self.tik_instance.Scalar(dtype="uint8", init_value=cur_core < self.left_over_ffts)

                    data_real_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.ub_size,), name="data_real_ub", scope=tik.scope_ubuf)
                    data_imag_ub = self.tik_instance.Tensor(dtype="float32", shape=(self.ub_size,), name="data_imag_ub", scope=tik.scope_ubuf)

                    with self.tik_instance.for_range(0, self.ffts_per_core + additional_frame) as cur_i:
                        cores_with_extra_frames = self.tik_instance.Scalar(dtype="int64", init_value=self.left_over_ffts)
                        cur_core_64 = self.tik_instance.Scalar(dtype="int64", init_value=cur_core)
                        self.tik_instance.scalar_min(cores_with_extra_frames, cur_core_64, cores_with_extra_frames)
                        cores_with_normal_frames = cur_core_64 - cores_with_extra_frames

                        fft_i = (cores_with_extra_frames * (self.ffts_per_core + 1)) + (cores_with_normal_frames * self.ffts_per_core) + cur_i
                        
                        self._count_fft_i(gm_cube, fft_i, data_real_ub, data_imag_ub, self.tmp_transpose_data[gm_offset_start:gm_offset_end])

                        if self.mode != "c2r":
                            self._transpose_and_write_real_imag(fft_i, data_real_ub, data_imag_ub)
                        else:
                            if (self.n % 8) == 0:
                                self.tik_instance.data_move(self.output_data[fft_i * self.n], data_real_ub, 0, 1, self.n * 4 // 32, 0, 0)
                            else:
                                self.tik_instance.data_move_pad(self.output_data[fft_i * self.n], data_real_ub, 1, self.n * 4, 0, 0)

                else:
                    self._fft1d_compute_large_case(cur_core, gm_cube)

                self.tik_instance.block_barrier(self.sync_workspace)


        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_data], outputs=[self.output_data], config={"enable_const_fold": True})
        return self.tik_instance


def fft1d(input, output, n=0, norm="backward", mode="c2c", forward=True, kernel_name="FFT1D"):
    obj = FFT1D(input, output, n, norm, mode, forward, kernel_name)
    return obj.fft1d_compute()
