#ifndef _SCENE_CUH_
#define _SCENE_CUH_

#include <iostream>
#include "utils.cuh"
#include "helper.cuh"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

/**
 * Represents the entire scene, with its geometry (as a list of triangles) and materials
 */
class Scene {
private:

public:
    /// @brief Device pointer to the buffer of all triangles in the scene
    Triangle* m_devTriangles;
    /// @brief Device pointer to the buffer of all materials in the scene
    Material* m_devMaterials;
    /// @brief Total number of triangles in the scene
    uint32_t  m_numTriangles;

    Scene() : m_devTriangles(NULL), m_devMaterials(NULL) { }

    /**
     * Load a scene from the specified .obj file
     * 
     * @param objFile the filename of the obj file to load
     * @returns true on success, false on failure
     */
    bool load(const char* objFile) {
        tinyobj::ObjReaderConfig readerConfig;
        readerConfig.triangulate = true;   // Triangulate input meshes
        readerConfig.vertex_color = false; // Don't care about vertex colors, we're going to use materials

        std::vector<Triangle> triangles;
        std::vector<Material> materials;

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

        // Triangles
        for (const auto shape : reader.GetShapes()) {
            for (size_t triangleIndex = 0; triangleIndex < shape.mesh.num_face_vertices.size(); ++triangleIndex) {
                Triangle tri;

                for (size_t i = 0; i < 3; ++i) {
                    auto index = shape.mesh.indices[triangleIndex * 3 + i].vertex_index;
                    auto vert = ((Vertex*)&tri) + i;

                    *vert = ((Vertex*)attrib.vertices.data())[index];
                }

                tri.materialIndex = shape.mesh.material_ids[triangleIndex];
                triangles.push_back(tri);
            }
        }

        // Materials
        for (const auto material : reader.GetMaterials()) {
            Material mat = {
                // Albedo
                RGBColor<float> {
                    material.diffuse[0],
                    material.diffuse[1],
                    material.diffuse[2]
                },
                // Emissivity
                RGBColor<float> {
                    material.emission[0],
                    material.emission[1],
                    material.emission[2]
                }
            };

            materials.push_back(mat);
        }

        // Allocate device memory and copy buffers
        m_numTriangles = triangles.size();
        checkCudaErrors(cudaMalloc((void**)&m_devTriangles, m_numTriangles * sizeof(Triangle)));
        checkCudaErrors(cudaMalloc((void**)&m_devMaterials, materials.size() * sizeof(Material)));

        checkCudaErrors(cudaMemcpy((void*)m_devTriangles, (void*)triangles.data(), m_numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void*)m_devMaterials, (void*)materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice));

        return true;
    }
    
    ~Scene() {
        // Deallocate device buffers
        if (m_devTriangles != NULL && m_devMaterials != NULL) {
            checkCudaErrors(cudaFree((void*)m_devTriangles));
            checkCudaErrors(cudaFree((void*)m_devMaterials));
        }
    }
};

#endif
