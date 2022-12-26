#if !defined(_BVH_CUH_) && !defined(NO_BVH)
#define _BVH_CUH_

#include <vector>
#include <queue>
#include <tuple>
#include <algorithm>
#include <inttypes.h>

#include "helper.cuh"
#include "utils.cuh"

/// @brief Identifies a leaf node
#define LEAF_NODE     0
/// @brief Identifies an internal node
#define INTERNAL_NODE 1

/**
 * An Axis-Aligned Bounding Box
 */
struct AABB {
    /// @brief Min and max coordinates of this aabb
    float3 lo, hi;

    /**
     * Expand this AABB to include the provided point
     * 
     * @param p the point for which to expand this aabb in order for it to be included
     */
    void expandToPoint(float3& p) {
        lo.x = min(lo.x, p.x);
        lo.y = min(lo.y, p.y);
        lo.z = min(lo.z, p.z);
        hi.x = max(hi.x, p.x);
        hi.y = max(hi.y, p.y);
        hi.z = max(hi.z, p.z);
    }
};

/**
 * A leaf node in the BVH, referencing a single triangle
 */
struct LeafNode {
    /// @brief The triangle index in the scene's triangle array
    uint32_t triangleIndex;
};

/**
 * An internal node in the BVH, referencing two sub-nodes
 */
struct InternalNode {
    /// @brief Index of the left sub-node in the BVH. The right one is implied being the next one due to how the structure is built
    uint32_t left;
};

/**
 * A generic node of the BVH. It can be either a LeafNode or an InternalNode
 */
struct Node {
    /// @brief The type of this node. This must be either LEAF_NODE or INTERNAL_NODE
    uint32_t type;
    /// @brief Bounding box of this node
    AABB aabb;
    /// @brief The node itself
    union {
        LeafNode leaf;
        InternalNode internal;
    } node;
};

/**
 * Represents a BVH acceleration structure
 */
class BVH {
private:
    /**
     * Expand a 10-bit unsigned integer into 30 bits by inserting
     * 2 zeros after each bit
     * 
     * @param v the integer to expand
     * @returns the expanded value
     */
    uint32_t expandBits(uint32_t v) {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;

        return v;
    }

    /**
     * Expand a 20-bit unsigned integer into 60 bits by inserting
     * 2 zeros after each bit
     * 
     * @param v the integer to expand
     * @returns the expanded value
     */
    uint64_t expandBits64(uint64_t v) {
        return ((uint64_t)(expandBits((uint32_t)((v >>  0) & 0x3FF)) <<  0)) +
               ((uint64_t)(expandBits((uint32_t)((v >> 10) & 0x3FF))) << 30);
    }

    /**
     * Calculate a 62-bit Morton code for the given point in world space
     * normalized w.r.t. the scene's AABB
     * 
     * @param p position with all dimensions in range [0, 1] for which to calculate the Morton code
     * @returns the calculated Morton code
     */
    uint64_t calculateMortonCode(double3& p) {
        uint64_t xx = expandBits64((uint64_t)(p.x * ((double)(1 << 20) - EPS)));
        uint64_t yy = expandBits64((uint64_t)(p.y * ((double)(1 << 20) - EPS)));
        uint64_t zz = expandBits64((uint64_t)(p.z * ((double)(1 << 20) - EPS)));

        return (xx << 2) + (yy << 1) + zz;
    }

    /**
     * Cross platform implementation of Count Leading Zeros for 64 bit values
     * 
     * @param v the value to count the leading zeros of
     * @returns the number of leading zeros in `v`
     */
    uint32_t clz(uint64_t v) {
        uint32_t lz = 0;
        while (lz < 64 && !(v & ((uint64_t)1 << ((uint64_t)63 - lz)))) ++lz;
        return lz;
    }

    /**
     * Find the split in a range of Morton codes based on the most significant bit that differs in the range
     * 
     * @param sortedMortonCodes array of tuples of (triangleIndex, mortonCode)
     * @param first first index in the range
     * @param last last index in the range, inclusive
     * @returns the index of the split position
     */
    uint32_t findSplit(std::tuple<uint32_t, uint64_t>* sortedMortonCodes, uint32_t first, uint32_t last) {
        auto firstCode = std::get<1>(sortedMortonCodes[first]);
        auto lastCode = std::get<1>(sortedMortonCodes[last]);

        if (firstCode == lastCode) {
            // The codes are the same, split range in half
            return (first + last) >> 1;
        }

        // Get the highest differing bit between firstCode and lastCode
        auto diffBit = 63 - clz(firstCode ^ lastCode);

        // Search the first code with said bit set to 1
        // TODO: this should ideally be done in O(log(n))
        for (uint32_t i = first + 1; i <= last; ++i) {
            auto code = std::get<1>(sortedMortonCodes[i]);
            if (code & ((uint64_t)1 << diffBit)) return i - 1;
        }

        // Unreachable
        throw;
    }
    
public:
    /// @brief Device pointer to the root node of the BVH
    Node* m_devRoot;
    /// @brief Number of nodes in the BVH
    uint32_t m_numNodes;
    
    /**
     * Build a BVH from a list of triangles and upload it to device memory
     * 
     * @param tris host buffer of all the triangles in the scene
     */
    BVH(std::vector<Triangle> tris) {
        // Compute the AABB containing the entire scene
        AABB sceneAABB = {
            make_float3(INFINITY, INFINITY, INFINITY),
            make_float3(-INFINITY, -INFINITY, -INFINITY)
        };
        for (auto tri : tris) {
            sceneAABB.expandToPoint(tri.v1);
            sceneAABB.expandToPoint(tri.v2);
            sceneAABB.expandToPoint(tri.v3);
        }

        // Sort triangle indices based on the Morton code of their AABB's centroid
        std::vector<std::tuple<uint32_t, uint64_t>> sortedTriangleIndices; // (triangleIndex, mortonCode)

        for (uint32_t i = 0; i < tris.size(); ++i) {
            // For each triangle, compute its AABB
            AABB triAABB = {
                make_float3(INFINITY, INFINITY, INFINITY),
                make_float3(-INFINITY, -INFINITY, -INFINITY)
            };
            triAABB.expandToPoint(tris[i].v1);
            triAABB.expandToPoint(tris[i].v2);
            triAABB.expandToPoint(tris[i].v3);

            // Then, calculate the Morton code of the centroid of said AABB
            auto centroid = triAABB.lo + (triAABB.hi - triAABB.lo)*0.5;
            auto centroidOffset = centroid - sceneAABB.lo;
            double3 centroidNormalized = {
                centroidOffset.x / (sceneAABB.hi - sceneAABB.lo).x,
                centroidOffset.y / (sceneAABB.hi - sceneAABB.lo).y,
                centroidOffset.z / (sceneAABB.hi - sceneAABB.lo).z,
            };
            auto morton = calculateMortonCode(centroidNormalized);

            sortedTriangleIndices.push_back(std::make_tuple(i, morton));
        }

        std::sort(
            sortedTriangleIndices.begin(),
            sortedTriangleIndices.end(),
            [](const std::tuple<uint32_t, uint64_t>& a, const std::tuple<uint32_t, uint64_t>& b) {
                return std::get<1>(a) < std::get<1>(b);
            }
        );

        // Build the BVH
        std::vector<Node> bvh;
        std::queue<std::tuple<uint32_t, uint32_t>> nodeQueue;
        nodeQueue.push(std::make_tuple((uint32_t)0, (uint32_t)(sortedTriangleIndices.size() - 1)));

        while (!nodeQueue.empty()) {
            Node node;
            
            // Pop the range
            uint32_t first, last;
            std::tie(first, last) = nodeQueue.front();
            nodeQueue.pop();

            if (first == last) {
                node.type = LEAF_NODE;
                std::tie(node.node.leaf.triangleIndex, std::ignore) = sortedTriangleIndices[first];
            } else {
                node.type = INTERNAL_NODE;
                node.node.internal.left = (uint32_t)(bvh.size() + nodeQueue.size() + 1);

                // Find split in range
                auto split = findSplit(sortedTriangleIndices.data(), first, last);

                // Schedule children ranges
                nodeQueue.push(std::make_tuple(first, split));
                nodeQueue.push(std::make_tuple(split + 1, last));
            }

            bvh.push_back(node);
        }

        // Compute the bounding boxes for every node (in last to first order)
        for (int32_t i = (int32_t)bvh.size() - 1; i >= 0; --i) {
            auto node = &bvh[i];

            switch (node->type) {
            case LEAF_NODE: {
                auto tri = &tris[node->node.leaf.triangleIndex];
                node->aabb = {
                    make_float3(INFINITY, INFINITY, INFINITY),
                    make_float3(-INFINITY, -INFINITY, -INFINITY)
                };
                node->aabb.expandToPoint(tri->v1);
                node->aabb.expandToPoint(tri->v2);
                node->aabb.expandToPoint(tri->v3);
                
                break;
            }
            
            case INTERNAL_NODE: {
                auto leftChild  = &bvh[node->node.internal.left];
                auto rightChild = leftChild + 1;

                node->aabb = leftChild->aabb;
                node->aabb.expandToPoint(rightChild->aabb.lo);
                node->aabb.expandToPoint(rightChild->aabb.hi);
                
                break;
            }

            default:
                // Unreachable
                throw;
            }
        }

        m_numNodes = bvh.size();

        // Transfer the BVH to the GPU
        checkCudaErrors(cudaMalloc((void**)&m_devRoot, m_numNodes * sizeof(Node)));
        checkCudaErrors(cudaMemcpy((void*)m_devRoot, (void*)bvh.data(), m_numNodes * sizeof(Node), cudaMemcpyHostToDevice));
    }

    ~BVH() {
        // Free device memory
        checkCudaErrors(cudaFree((void*)m_devRoot));
    }
};

#endif
