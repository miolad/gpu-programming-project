#include <iostream>
#include <chrono>
#include <thread>
#include <inttypes.h>

#include "helper.cuh"
#include "framebuffer.cuh"
#include "scene.cuh"
#include "utils.cuh"
#include "camera.cuh"
#include "cudaviewer.h"
#include "path_tracing.cuh"

int main(int argc, char* argv[]) {
    // Parse command line arguments
    if (argc != 2) {
        std::cerr << "USAGE: " << argv[0] << " <scene obj file>" << std::endl;
        exit(EXIT_FAILURE);
    }
    const char* sceneObjFilename = argv[1];

    // Create a Vulkan shared framebuffer
    Framebuffer fb(make_int2(RES_X, RES_Y));
    
    // Load scene from obj
    std::cout << "Loading scene (" << sceneObjFilename << ")...";
    std::cout.flush();
    
    Scene scene;
    if (!scene.load(sceneObjFilename, make_int2(RES_X, RES_Y))) {
        exit(EXIT_FAILURE);
    }

    std::cout << "DONE (" << scene.m_numTriangles << " triangles";
#ifndef NO_NEXT_EVENT_ESTIMATION
    std::cout << ", " << scene.m_numLights << " lights)" << std::endl;
#else
    std::cout << ")" << std::endl;
#endif

    // Initialize viewer
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
    void* viewerCtx = viewer::init(fb.m_shareableHandle, fb.m_size, RES_X, RES_Y, (uint8_t*)&deviceProps.uuid.bytes);
    
    // Render main loop
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreate(&event));
    uint32_t batch = 0;
    auto gridSize = dim3(
        ceilf((float)RES_X / 16.0),
        ceilf((float)RES_Y / 16.0)
    );
    auto blockSize = dim3(
        16, 16
    );
    
#ifndef NO_BVH
    uint32_t bvhCachedNodesNum = min(scene.m_bvh->m_numNodes, (uint32_t)(USE_SHARED_MEMORY_AMOUNT / sizeof(Node)));
#endif

    uint32_t sharedMemoryAmount =
#ifndef NO_BVH
        bvhCachedNodesNum * sizeof(Node)
#else
        0
#endif
    ;

    while (true) {
        checkCudaErrors(cudaEventRecord(event));
        pathTrace<<<gridSize, blockSize, sharedMemoryAmount>>>(
            scene.m_devTriangles,
            scene.m_devMaterials,
            scene.m_numTriangles,
#ifndef NO_BVH
            scene.m_bvh->m_devRoot,
            bvhCachedNodesNum,
#endif
#ifndef NO_NEXT_EVENT_ESTIMATION
            scene.m_devLights,
            scene.m_numLights,
#endif
            *scene.m_camera,
            batch++,
            fb.m_devPtr
        );
        checkCudaErrors(cudaGetLastError());

        // Run the viewer's event loop while we wait to resubmit the CUDA kernel
        bool shouldClose = false;
        while (!(shouldClose = viewer::run_event_loop(viewerCtx)) && cudaEventQuery(event) != cudaSuccess) {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }

        if (shouldClose) break;
    }

    // Write framebuffer to PNG file
    fb.saveToPNG();

    checkCudaErrors(cudaEventDestroy(event));

    // Deinit viewer
    viewer::deinit(viewerCtx);

    return EXIT_SUCCESS;
}
