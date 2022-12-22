#ifndef _SCENE_CUH_
#define _SCENE_CUH_

#include <iostream>
#include <inttypes.h>
#include "utils.cuh"
#include "helper.cuh"
#include "camera.cuh"

#ifndef NO_BVH
#include "bvh.cuh"
#endif

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

/**
 * Computes normals for all the passed triangles
 * 
 * @param tris device buffer containing all the triangles in the scene
 * @param triNum number of triangles pointed to by `tris`
 */
__global__ void computeTriangleNormals(Triangle* tris, uint32_t triNum) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= triNum) return;

    auto tri = tris[tid];
    tris[tid].normal = normalize(cross(
        tri.v2 - tri.v1,
        tri.v3 - tri.v1
    ));
}

/**
 * Represents the entire scene, with its geometry (as a list of triangles) and materials
 */
class Scene {
public:
    /// @brief Device pointer to the buffer of all triangles in the scene
    Triangle* m_devTriangles;
    /// @brief Device pointer to the buffer of all materials in the scene
    Material* m_devMaterials;
    /// @brief Total number of triangles in the scene
    uint32_t  m_numTriangles;
    /// @brief Camera associated to the scene. Loading an obj also initializes the camera
    Camera* m_camera;

#ifndef NO_NEXT_EVENT_ESTIMATION
    /// @brief List of all indices into `m_devTriangles` of triangle with an emissive material
    uint32_t* m_devLights;
    /// @brief Number of indices in `m_devLights`
    uint32_t m_numLights;
#endif

    Scene() : m_devTriangles(NULL), m_devMaterials(NULL), m_camera(NULL) { }

    /**
     * Load a scene from the specified .obj file
     * 
     * @param objFile the filename of the obj file to load
     * @param resolution the full framebuffer resolution, used to initialize the camera
     * @returns true on success, false on failure
     */
    bool load(const char* objFile, int2 resolution) {
        tinyobj::ObjReaderConfig readerConfig;
        readerConfig.triangulate = true;   // Triangulate input meshes
        readerConfig.vertex_color = false; // Don't care about vertex colors, we're going to use materials

        std::vector<Triangle> triangles;
        std::vector<Material> materials;

#ifndef NO_NEXT_EVENT_ESTIMATION
        std::vector<uint32_t> lights;
#endif

        tinyobj::ObjReader reader;
        if (!reader.ParseFromFile(objFile, readerConfig)) {
            std::cerr << "Error reading scene from " << objFile << std::endl;
            return false;
        }

        if (!reader.Error().empty()) {
            std::cerr << "Error reading scene from " << objFile << ": " << reader.Error() << std::endl;
            return false;
        }

        if (!reader.Warning().empty()) {
            std::cerr << "Warning reading scene from " << objFile << ": " << reader.Warning() << std::endl;
        }

        auto attrib = reader.GetAttrib();

        // Materials
        for (const auto material : reader.GetMaterials()) {
            Material mat = {
                // Albedo
                {
                    material.diffuse[0],
                    material.diffuse[1],
                    material.diffuse[2]
                },
                // Emissivity
                {
                    material.emission[0],
                    material.emission[1],
                    material.emission[2]
                }
            };

            materials.push_back(mat);
        }

        // Triangles
        for (const auto shape : reader.GetShapes()) {
            if (shape.name == "cameraParameters" && shape.points.indices.size() == 4) {
                // Load camera parameters
                float3 position = ((float3*)attrib.vertices.data())[shape.points.indices[0].vertex_index];
                float3 up       = ((float3*)attrib.vertices.data())[shape.points.indices[1].vertex_index];
                float3 view     = ((float3*)attrib.vertices.data())[shape.points.indices[2].vertex_index];
                float  hfov     = attrib.vertices[shape.points.indices[3].vertex_index * 3];

                m_camera = new Camera(resolution, position, up, view, hfov);
            }
            
            for (size_t triangleIndex = 0; triangleIndex < shape.mesh.num_face_vertices.size(); ++triangleIndex) {
                Triangle tri;

                for (size_t i = 0; i < 3; ++i) {
                    auto index = shape.mesh.indices[triangleIndex * 3 + i].vertex_index;
                    auto vert = ((float3*)&tri) + i;

                    *vert = ((float3*)attrib.vertices.data())[index];
                }

                tri.materialIndex = shape.mesh.material_ids[triangleIndex];

#ifndef NO_NEXT_EVENT_ESTIMATION
                auto emissivity = materials[tri.materialIndex].emissivity;
                if (max(emissivity.x, max(emissivity.y, emissivity.z)) > EPS) {
                    // Triangle is emissive, add to list of lights
                    lights.push_back(triangles.size());
                }
#endif

                triangles.push_back(tri);
            }
        }

        if (m_camera == NULL) {
            std::cerr << "Camera parameters not found or invalid in scene obj, configuring generic camera" << std::endl;
            m_camera = new Camera(resolution, make_float3(0.0, 0.0, 0.0), make_float3(0.0, 1.0, 0.0), make_float3(0.0, 0.0, -1.0), 65.0);
        }

        // Allocate device memory and copy buffers
        m_numTriangles = triangles.size();
        checkCudaErrors(cudaMalloc((void**)&m_devTriangles, m_numTriangles * sizeof(Triangle)));
        checkCudaErrors(cudaMalloc((void**)&m_devMaterials, materials.size() * sizeof(Material)));

        checkCudaErrors(cudaMemcpy((void*)m_devTriangles, (void*)triangles.data(), m_numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void*)m_devMaterials, (void*)materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

#ifndef NO_NEXT_EVENT_ESTIMATION
        m_numLights = lights.size();
        checkCudaErrors(cudaMalloc((void**)&m_devLights, m_numLights * sizeof(uint32_t)));
        checkCudaErrors(cudaMemcpy((void*)m_devLights, (void*)lights.data(), m_numLights * sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif

        // Compute triangle normals
        computeTriangleNormals<<<ceil((float)m_numTriangles / 256.0), 256>>>(m_devTriangles, m_numTriangles);

#ifndef NO_BVH
        // Build BVH
        BVH bvh(triangles);
#endif

        return true;
    }
    
    ~Scene() {
        // Deallocate device buffers
        if (m_devTriangles != NULL && m_devMaterials != NULL) {
            checkCudaErrors(cudaFree((void*)m_devTriangles));
        }
        if (m_devMaterials != NULL) {
            checkCudaErrors(cudaFree((void*)m_devMaterials));
        }

        // Deallocate camera
        if (m_camera != NULL) {
            delete m_camera;
        }
    }
};

#endif
