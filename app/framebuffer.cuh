#ifndef _FRAMEBUFFER_CUH_
#define _FRAMEBUFFER_CUH_

#include <cuda.h>
#include "helper.cuh"

#define ROUND_UP_TO_GRANULARITY(x, n) (((x + n - 1) / n) * n)

#if (defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)) && !defined(USE_ZERO_COPY_MEMORY)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <sddl.h>
#include <winternl.h>
#endif

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
            printf("IPC failure: getDefaultSecurityDescriptor Failed! (%d)\n",
                    GetLastError());
            }

            InitializeObjectAttributes(&objAttributes, NULL, 0, NULL, secDesc);
            objAttributesConfigured = true;
        }

        prop->win32HandleMetaData = &objAttributes;
    }
#endif
#endif
    
public:
    size_t m_size;
    uint32_t* m_devPtr;
    void* m_shareableHandle;

    Framebuffer(int2 resolution) {
        // Allocate the framebuffer memory in a Vulkan shared memory pool
#ifdef USE_ZERO_COPY_MEMORY
        // Align to a ridiculously large number to stay on the safe side
        m_size = ROUND_UP_TO_GRANULARITY(resolution.x * resolution.y * 4, 4096);

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
        m_size = ROUND_UP_TO_GRANULARITY(resolution.x + resolution.y * 4, granularity);

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
