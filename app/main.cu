#include <stdio.h>
#include <inttypes.h>
#include <cuda.h>
#include <assert.h>
#include <cudaviewer.h>

#if (defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)) && !defined(USE_ZERO_COPY_MEMORY)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <sddl.h>
#include <winternl.h>
#endif

#if !defined(USE_ZERO_COPY_MEMORY)
static const char *_cudaGetErrorEnum(CUresult error) {
  static char unknown[] = "<unknown>";
  const char *ret = NULL;
  cuGetErrorName(error, &ret);
  return ret ? ret : unknown;
}

#ifdef __DRIVER_TYPES_H__
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#endif

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}
#endif

#define ROUND_UP_TO_GRANULARITY(x, n) (((x + n - 1) / n) * n)

#define RES_X 800
#define RES_Y 600

#if !defined(USE_ZERO_COPY_MEMORY)
#if defined(__linux__)
CUmemAllocationHandleType shareable_mem_handle_type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
CUmemAllocationHandleType shareable_mem_handle_type = CU_MEM_HANDLE_TYPE_WIN32;
#endif
#endif

#if !defined(USE_ZERO_COPY_MEMORY)
void get_default_security_descriptor(CUmemAllocationProp* prop) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
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
#endif
}
#endif

__global__ void fill_image(uint32_t* img, uint8_t b) {
    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= RES_X || y >= RES_Y) return;

    uint8_t r = (uint8_t)(((float)x * 255.0) / (float)RES_X);
    uint8_t g = (uint8_t)(((float)y * 255.0) / (float)RES_Y);

    uint64_t index = x + y * RES_X;
    img[index] = (uint32_t)r + ((uint32_t)g << 8) + ((uint32_t)b << 16);
}

int main() {
    // Allocate shareable memory for the framebuffer
#if defined(USE_ZERO_COPY_MEMORY)
    void* d_ptr, * mem_shareable_handle;
    size_t framebuffer_size = ROUND_UP_TO_GRANULARITY(RES_X * RES_Y * 4, 4096); // Align the size to a ridiculously large number just to be on the safe side
    cudaHostAlloc(&mem_shareable_handle, framebuffer_size, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_ptr, mem_shareable_handle, 0);
#else
    // Initialize driver API
    checkCudaErrors(cuInit(0));
    
    CUmemAllocationProp alloc_prop = {};
    alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    alloc_prop.location.id = 0;
    alloc_prop.requestedHandleTypes = shareable_mem_handle_type;
    get_default_security_descriptor(&alloc_prop);

    size_t granularity;
    checkCudaErrors(cuMemGetAllocationGranularity(&granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    // 8-bit per channel RGBA (A is mostly for better coalescing of global memory accesses)
    size_t framebuffer_size = ROUND_UP_TO_GRANULARITY((RES_X * RES_Y * 4), granularity);

    CUdeviceptr d_ptr;
    checkCudaErrors(cuMemAddressReserve(&d_ptr, framebuffer_size, granularity, 0, 0));
    
    CUmemGenericAllocationHandle allocation_handle;
    checkCudaErrors(cuMemCreate(&allocation_handle, framebuffer_size, &alloc_prop, 0));

    void* mem_shareable_handle;
    checkCudaErrors(cuMemExportToShareableHandle(&mem_shareable_handle, allocation_handle, shareable_mem_handle_type, 0));

    checkCudaErrors(cuMemMap(d_ptr, framebuffer_size, 0, allocation_handle, 0));
    checkCudaErrors(cuMemRelease(allocation_handle));

    CUmemAccessDesc access_descriptor = {};
    access_descriptor.location.id = 0;
    access_descriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_descriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    checkCudaErrors(cuMemSetAccess(d_ptr, framebuffer_size, &access_descriptor, 1));
#endif

    // Initialize viewer
    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);
    void* viewer_ctx = viewer::init(mem_shareable_handle, framebuffer_size, RES_X, RES_Y, (uint8_t*)&device_props.uuid.bytes);
    
    // "Render" into the framebuffer
    cudaEvent_t event;
    cudaEventCreate(&event);
    
    uint8_t b = 0;
    while (true) {
        cudaEventRecord(event);
        fill_image<<<dim3(ceil(float(RES_X)/32.0), ceil(float(RES_Y)/32.0)), dim3(32, 32)>>>((uint32_t*)d_ptr, b++);

        // Run the viewer's event loop while we wait to resubmit the CUDA kernel
        bool should_close = false;
        while (!(should_close = viewer::run_event_loop(viewer_ctx)) && cudaEventQuery(event) != cudaSuccess);

        if (should_close) break;
    }

    cudaEventDestroy(event);

    // Deinit viewer
    viewer::deinit(viewer_ctx);

    // Free memory
#if defined(USE_ZERO_COPY_MEMORY)
    cudaFreeHost(mem_shareable_handle);
#else
    checkCudaErrors(cuMemUnmap(d_ptr, framebuffer_size));
    checkCudaErrors(cuMemAddressFree(d_ptr, framebuffer_size));
#endif
    
    return EXIT_SUCCESS;
}
