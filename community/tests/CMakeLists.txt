


if(AICPU_UT)
    include(aicpu_ut.cmake)
elseif(PROTO_UT)
    include(op_proto_ut.cmake)
elseif(TILING_UT)
    include(tiling_ut.cmake)
else()
    include(aicpu_ut.cmake)
    include(op_proto_ut.cmake)
    include(tiling_ut.cmake)
endif()