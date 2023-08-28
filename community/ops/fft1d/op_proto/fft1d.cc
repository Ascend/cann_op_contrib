#include "inc/fft1d.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(FFT1DInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(FFT1D, FFT1DVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FFT1D, FFT1DInferShape);
VERIFY_FUNC_REG(FFT1D, FFT1DVerify);

}  // namespace ge
