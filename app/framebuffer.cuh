#ifndef _FRAMEBUFFER_CUH_
#define _FRAMEBUFFER_CUH_

#include <iostream>
#include <cuda.h>
#include "helper.cuh"
#include "utils.cuh"
#include "lodepng.h"

#define ROUND_UP_TO_GRANULARITY(x, n) (((x + n - 1) / n) * n)

#if (defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)) && !defined(USE_ZERO_COPY_MEMORY)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <sddl.h>
#include <winternl.h>
#endif

/**
 * Performs gamma correction on the framebuffer and compresses it into 8-bit, RGB pixels
 * 
 * @param fb input framebuffer
 * @param out output buffer to store the result into
 */
__global__ void prepareFbForOutput(const float3* __restrict__ fb, uint8_t* __restrict__ out) {
    uint32_t x = threadIdx.x + blockIdx.x*blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y*blockDim.y;
    uint32_t pixelIndex = x + y*RES_X;

    if (x >= RES_X || y >= RES_Y) return;

    float3 pixel = fb[pixelIndex];
    
    // Perform gamma correction
    pixel.x = powf(pixel.x, 1.0f / 2.2f);
    pixel.y = powf(pixel.y, 1.0f / 2.2f);
    pixel.z = powf(pixel.z, 1.0f / 2.2f);

    // Write to 8-bit channels
    out[3*pixelIndex + 0] = (uint8_t)clamp(pixel.x * 256.0f, 0.0f, 255.0f);
    out[3*pixelIndex + 1] = (uint8_t)clamp(pixel.y * 256.0f, 0.0f, 255.0f);
    out[3*pixelIndex + 2] = (uint8_t)clamp(pixel.z * 256.0f, 0.0f, 255.0f);
}

/**
 * Simple abstraction over the creation of shareable memory for the app's framebuffer
 */
class Framebuffer {
private:
#ifndef USE_ZERO_COPY_MEMORY
    static const CUmemAllocationHandleType s_shareableMemHandleType =
#ifdef __linux__
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
    CU_MEM_HANDLE_TYPE_WIN32;
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    static void getWinDefaultSecurityDescriptor(CUmemAllocationProp* prop) {
        static const char sddl[] = "D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)";
        static OBJECT_ATTRIBUTES objAttributes;
        static bool objAttributesConfigured = false;

        if (!objAttributesConfigured) {
            PSECURITY_DESCRIPTOR secDesc;
            BOOL result = ConvertStringSecurityDescriptorToSecurityDescriptorA(
                sddl, SDDL_REVISION_1, &secDesc, NULL);
            if (result == 0) {
                std::cerr << "IPC failure: getWinDefaultSecurityDescriptor failed! " << GetLastError() << std::endl;
            }

            InitializeObjectAttributes(&objAttributes, NULL, 0, NULL, secDesc);
            objAttributesConfigured = true;
        }

        prop->win32HandleMetaData = &objAttributes;
    }
#endif
#endif
    
public:
    /// @brief Size of this framebuffer in bytes
    size_t m_size;
    /// @brief Device accessible pointer to the framebuffer
    float3* m_devPtr;
    /// @brief Opaque shareable handle to the framebuffer
    void* m_shareableHandle;

    Framebuffer(int2 resolution) {
        // Allocate the framebuffer memory in a Vulkan shared memory pool
#ifdef USE_ZERO_COPY_MEMORY
        // Align to a ridiculously large number to stay on the safe side
        m_size = ROUND_UP_TO_GRANULARITY(resolution.x * resolution.y * sizeof(float3), 4096);

        checkCudaErrors(cudaHostAlloc(&m_shareableHandle, m_size, cudaHostAllocMapped));
        checkCudaErrors(cudaHostGetDevicePointer((void**)&m_devPtr, m_shareableHandle, 0));
#else
        // Initialize driver API
        checkCudaErrors(cuInit(0));

        CUmemAllocationProp allocProp = {};
        allocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        allocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        allocProp.location.id = 0;
        allocProp.requestedHandleTypes = s_shareableMemHandleType;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        getWinDefaultSecurityDescriptor(&allocProp);
#endif

        size_t granularity;
        checkCudaErrors(cuMemGetAllocationGranularity(&granularity, &allocProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        m_size = ROUND_UP_TO_GRANULARITY(resolution.x * resolution.y * sizeof(float3), granularity);

        CUmemGenericAllocationHandle allocationHandle;
        checkCudaErrors(cuMemAddressReserve((CUdeviceptr*)&m_devPtr, m_size, granularity, 0, 0));
        checkCudaErrors(cuMemCreate(&allocationHandle, m_size, &allocProp, 0));
        checkCudaErrors(cuMemExportToShareableHandle(&m_shareableHandle, allocationHandle, s_shareableMemHandleType, 0));

        checkCudaErrors(cuMemMap((CUdeviceptr)m_devPtr, m_size, 0, allocationHandle, 0));
        checkCudaErrors(cuMemRelease(allocationHandle)); // This won't actually free the memory until it's mapped

        CUmemAccessDesc accessDescriptor = {};
        accessDescriptor.location.id = 0;
        accessDescriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        
        checkCudaErrors(cuMemSetAccess((CUdeviceptr)m_devPtr, m_size, &accessDescriptor, 1));
#endif

        // Clear the framebuffer
        checkCudaErrors(cudaMemset((void*)m_devPtr, 0, m_size));
    }

    /**
     * Perform gamma correction to sRGB, then save the current contents of the framebuffer to a PNG image
     */
    void saveToPNG() {
        uint8_t* h_tmpBuf, * d_tmpBuf;
        
        // Allocate temporary buffer in host and device memory
        checkCudaErrors(cudaMallocHost((void**)&h_tmpBuf, RES_X*RES_Y*3));
        checkCudaErrors(cudaMalloc((void**)&d_tmpBuf, RES_X*RES_Y*3));

        // Prepare framebuffer for output
        prepareFbForOutput<<<dim3(ceilf((float)RES_X / 32.0f), ceilf((float)RES_Y / 32.0f)), dim3(32, 32)>>>(m_devPtr, d_tmpBuf);

        // Transfer result to host memory (this also serves as synchronization point)
        checkCudaErrors(cudaMemcpy((void*)h_tmpBuf, (void*)d_tmpBuf, RES_X*RES_Y*3, cudaMemcpyDeviceToHost));

        // Encode PNG and write to file
        auto err = lodepng::encode("out.png", h_tmpBuf, RES_X, RES_Y, LCT_RGB, 8);
        if (err) std::cout << "PNG encode error: " << lodepng_error_text(err) << std::endl;
        else     std::cout << "Written output to out.png" << std::endl;

        // Free temporary buffers
        checkCudaErrors(cudaFreeHost((void*)h_tmpBuf));
        checkCudaErrors(cudaFree((void*)d_tmpBuf));
    }

    ~Framebuffer() {
        // Free the framebuffer memory
#ifdef USE_ZERO_COPY_MEMORY
        checkCudaErrors(cudaFreeHost(m_shareableHandle));
#else
        checkCudaErrors(cuMemUnmap((CUdeviceptr)m_devPtr, m_size));
        checkCudaErrors(cuMemAddressFree((CUdeviceptr)m_devPtr, m_size));
#endif
    }
};

#endif
