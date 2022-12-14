#ifndef _HELPER_CUH_
#define _HELPER_CUH_

#include <iostream>
#include <cuda.h>

#if !defined(USE_ZERO_COPY_MEMORY)
static const char *_cudaGetErrorEnum(CUresult error) {
    static char unknown[] = "<unknown>";
    const char *ret = NULL;
    cuGetErrorName(error, &ret);
    return ret ? ret : unknown;
}
#endif

static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) <<
            "(" << _cudaGetErrorEnum(result) << ") \"" << func << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }
}

#ifdef __DRIVER_TYPES_H__
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#endif

#endif
