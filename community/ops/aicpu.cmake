
# --------------------------------aicpu---------------------------------------
if (MINRC)
    set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
else()
    set(CMAKE_CXX_COMPILER $ENV{ASCEND_AICPU_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
    set(CMAKE_C_COMPILER   $ENV{ASCEND_AICPU_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc)
endif()


set(CMAKE_CXX_STANDARD 11)

file(GLOB AICPU_SRC ${CANN_ROOT_DIR}/community/ops/**/aicpu/impl/*.cc
    ${CANN_ROOT_DIR}/community/common/utils/allocator_utils.cc
    ${CANN_ROOT_DIR}/community/common/utils/bcast.cc
    ${CANN_ROOT_DIR}/community/common/utils/broadcast_iterator.cc
    ${CANN_ROOT_DIR}/community/common/utils/eigen_tensor.cc
    ${CANN_ROOT_DIR}/community/common/utils/kernel_util.cc
    ${CANN_ROOT_DIR}/community/common/utils/range_sampler.cc
    ${CANN_ROOT_DIR}/community/common/utils/sampling_kernels.cc
    ${CANN_ROOT_DIR}/community/common/utils/sparse_group.cc
    ${CANN_ROOT_DIR}/community/common/utils/sparse_tensor.cc
)


if(EXISTS $ENV{ASCEND_AICPU_PATH}/opp/op_impl/built-in/aicpu/aicpu_kernel)
    set(AICPU_OPP_ENV $ENV{ASCEND_AICPU_PATH}/opp/op_impl/built-in/aicpu/aicpu_kernel)
else()
    set(AICPU_OPP_ENV $ENV{ASCEND_AICPU_PATH}/opp/built-in/op_impl/aicpu/aicpu_kernel)
endif()

set(AICPU_INCLUDE ${AICPU_OPP_ENV}/inc)
set(AICPU_INC
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/third_party/graphengine/inc
    ${METADEF_DIR}/third_party/graphengine/inc/external
    ${METADEF_DIR}/third_party/fwkacllib/inc
    ${CANN_ROOT_DIR}/community/common
    ${CANN_ROOT_DIR}/community/common/inc
    ${CANN_ROOT_DIR}/community/common/src
    ${AICPU_INCLUDE}
    ${EIGEN_INCLUDE}
    ${C_SEC_INCLUDE}
)


add_library(cust_aicpu_kernels SHARED
    ${AICPU_SRC}
)
add_dependencies(cust_aicpu_kernels eigen c_sec)
target_include_directories(cust_aicpu_kernels PRIVATE
    ${AICPU_INC}
)


if("${AICPU_SOC_VERSION}" STREQUAL "")
    set(AICPU_SOC_VERSION "Ascend910")
else()
    set(AICPU_SOC_VERSION $ENV{AICPU_SOC_VERSION})
endif()

if(EXISTS "${AICPU_OPP_ENV}/lib/${AICPU_SOC_VERSION}/libascend_protobuf.a")
    target_link_libraries(cust_aicpu_kernels PRIVATE
        -Wl,--whole-archive
        ${AICPU_OPP_ENV}/lib/${AICPU_SOC_VERSION}/libascend_protobuf.a
        -Wl,--no-whole-archive
        -s
        -Wl,-Bsymbolic
        -Wl,--exclude-libs=libascend_protobuf.a
    )
endif()

if(EXISTS "${AICPU_OPP_ENV}/lib/${AICPU_SOC_VERSION}/libcpu_kernels_context.a")
    target_link_libraries(cust_aicpu_kernels PRIVATE
        -Wl,--whole-archive
        ${AICPU_OPP_ENV}/lib/${AICPU_SOC_VERSION}/libcpu_kernels_context.a
        -Wl,--no-whole-archive
    )
else()
    if(EXISTS "${AICPU_OPP_ENV}/lib/libcpu_kernels_context.a")
        target_link_libraries(cust_aicpu_kernels PRIVATE
            -Wl,--whole-archive
            ${AICPU_OPP_ENV}/lib/libcpu_kernels_context.a
            -Wl,--no-whole-archive
        )
    elseif(EXISTS "${AICPU_OPP_ENV}/lib/device/libcpu_kernels_context.so")
        target_link_libraries(cust_aicpu_kernels PRIVATE
            -Wl,--whole-archive
            ${AICPU_OPP_ENV}/lib/device/libcpu_kernels_context.so
            -Wl,--no-whole-archive
        )
    endif()
endif()


set(AICPU_PATH "${INSTALL_DIR}/community/cpu/aicpu_kernel/impl")
cann_install(
    TARGET      cust_aicpu_kernels
    FILES       $<TARGET_FILE:cust_aicpu_kernels>
    DESTINATION "${AICPU_PATH}"
)
