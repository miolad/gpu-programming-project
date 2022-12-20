#include <iostream>
#include <inttypes.h>

#include "helper.cuh"
#include "framebuffer.cuh"
#include "scene.cuh"
#include "utils.cuh"
#include "camera.cuh"
#include "cudaviewer.h"
#include "path_tracing.cuh"

int main() {
    // Create a Vulkan shared framebuffer
    Framebuffer fb(make_int2(RES_X, RES_Y));
    
    // Load scene from obj
    std::cout << "Loading scene (" << SCENE << ")...";
    std::cout.flush();
    
    Scene scene;
    if (!scene.load(SCENE, make_int2(RES_X, RES_Y))) {
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE (" << scene.m_numTriangles << " triangles)" << std::endl;

    // Initialize viewer
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
    void* viewerCtx = viewer::init(fb.m_shareableHandle, fb.m_size, RES_X, RES_Y, (uint8_t*)&deviceProps.uuid.bytes);
    
    // Render main loop
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreate(&event));
    uint32_t batch = 0;
    auto gridSize = dim3(
        ceilf((float)RES_X / 32.0),
        ceilf((float)RES_Y / 16.0)
    );
    auto blockSize = dim3(
        32, 16
    );
    
    while (true) {
        checkCudaErrors(cudaEventRecord(event));
        pathTrace<<<gridSize, blockSize>>>(
            scene.m_devTriangles,
            scene.m_devMaterials,
            scene.m_numTriangles,
            *scene.m_camera,
            batch++,
            fb.m_devPtr
        );
        checkCudaErrors(cudaGetLastError());

        // Run the viewer's event loop while we wait to resubmit the CUDA kernel
        // bool shouldClose = false;
        // while (!(shouldClose = viewer::run_event_loop(viewerCtx)) && cudaEventQuery(event) != cudaSuccess);
        checkCudaErrors(cudaEventSynchronize(event));
        if (viewer::run_event_loop(viewerCtx)) break;

        // if (shouldClose) break;
    }

    checkCudaErrors(cudaEventDestroy(event));

    // Deinit viewer
    viewer::deinit(viewerCtx);

    return EXIT_SUCCESS;
}
