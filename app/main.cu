#include <inttypes.h>

#include "helper.cuh"
#include "framebuffer.cuh"
#include "scene.cuh"
#include "utils.cuh"
#include "camera.cuh"
#include "cudaviewer.h"

#define RES_X 800
#define RES_Y 600
#define SCENE "scenes/cornell_box.obj"

__global__ void fill_image(RGBColor<uint8_t>* img, uint8_t b) {
    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= RES_X || y >= RES_Y) return;

    uint8_t r = (uint8_t)(((float)x * 255.0) / (float)RES_X);
    uint8_t g = (uint8_t)(((float)y * 255.0) / (float)RES_Y);

    uint32_t index = x + y * RES_X;
    RGBColor<uint8_t> result = {
        r, g, b
    };
    // This wild way to write to the framebuffer greatly increases performance
    // with USE_ZERO_COPY_MEMORY on discrete GPU systems (instead of just img[index] = result),
    // apparently because it writes one u32 instead of 3 u8s to global memory (in RAM)
    ((uint32_t*)img)[index] = reinterpret_cast<uint32_t const&>(result);
}

int main() {
    // Create a Vulkan shared framebuffer
    Framebuffer fb(make_int2(RES_X, RES_Y));
    
    // Load scene from obj
    Scene scene;
    if (!scene.load(SCENE, make_int2(RES_X, RES_Y))) {
        exit(EXIT_FAILURE);
    }

    std::cout << "Loaded scene (" << SCENE << "): " << scene.m_numTriangles << " triangles" << std::endl;

    // Initialize viewer
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
    void* viewerCtx = viewer::init(fb.m_shareableHandle, fb.m_size, RES_X, RES_Y, (uint8_t*)&deviceProps.uuid.bytes);
    
    // Render main loop
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreate(&event));
    
    uint8_t b = 0;
    while (true) {
        checkCudaErrors(cudaEventRecord(event));
        fill_image<<<dim3(ceil(float(RES_X)/32.0), ceil(float(RES_Y)/32.0)), dim3(32, 32)>>>(fb.m_devPtr, b++);

        // Run the viewer's event loop while we wait to resubmit the CUDA kernel
        bool shouldClose = false;
        while (!(shouldClose = viewer::run_event_loop(viewerCtx)) && cudaEventQuery(event) != cudaSuccess);

        if (shouldClose) break;
    }

    checkCudaErrors(cudaEventDestroy(event));

    // Deinit viewer
    viewer::deinit(viewerCtx);

    return EXIT_SUCCESS;
}
