#include "Static/KTruss/KTruss.cuh"
#include "KTrussOperators.cuh"
#include "KTrussSupport.cuh"

#include <iostream>

using namespace std;

namespace hornets_nest {

void kTrussOneIteration(HornetGraph& hornet,
                        triangle_t*  __restrict__ output_triangles,
                        int threads_per_block,
                        int number_blocks,
                        int shifter,
                        int thread_blocks,
                        int blockdim,
                        HostDeviceVar<KTrussData>& hd_data);

//==============================================================================

KTruss::KTruss(HornetGraph& hornet) : StaticAlgorithm(hornet) {
    hd_data().active_queue.initialize(hornet);
    originalNE = hornet.nE();
    originalNV = hornet.nV();

}

KTruss::~KTruss() {
    release();
}

void KTruss::setInitParameters(int tsp, int nbl, int shifter,
                               int blocks, int sps) {
    hd_data().tsp     = tsp;
    hd_data().nbl     = nbl;
    hd_data().shifter = shifter;
    hd_data().blocks  = blocks;
    hd_data().sps     = sps;
}

void KTruss::init(){
    gpu::allocate(hd_data().is_active,            originalNV);
    gpu::allocate(hd_data().offset_array,         originalNV + 1);
    gpu::allocate(hd_data().triangles_per_vertex, originalNV);
    gpu::allocate(hd_data().triangles_per_edge,   originalNE);
    gpu::allocate(hd_data().src,                  originalNE);
    gpu::allocate(hd_data().dst,                  originalNE);
    gpu::allocate(hd_data().counter,              1);
    gpu::allocate(hd_data().active_vertices,      1);
    reset();
}

void KTruss::createOffSetArray(){

    gpu::memsetZero(hd_data().offset_array, originalNV+1);

    int *tempSize;
    gpu::allocate(tempSize, originalNV+1);

    forAllVertices(hornet, getVertexSizes {tempSize});


    void     *d_temp_storage = NULL; size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, tempSize, hd_data().offset_array+1, originalNV);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, tempSize, hd_data().offset_array+1, originalNV);
    cudaFree(d_temp_storage);  


    gpu::free(tempSize);
}

void KTruss::copyOffsetArrayHost(const vert_t* host_offset_array) {
    cudaMemcpy(hd_data().offset_array,host_offset_array,(originalNV + 1)*sizeof(vert_t), cudaMemcpyHostToDevice);
}

void KTruss::copyOffsetArrayDevice(vert_t* device_offset_array){
    cudaMemcpy(hd_data().offset_array,device_offset_array,(originalNV + 1)*sizeof(vert_t), cudaMemcpyDeviceToDevice);
}

vert_t KTruss::getMaxK() {
    return hd_data().max_K;
}

void KTruss::sortHornet(){
        forAllVertices(hornet, SimpleBubbleSort {});
}

//==============================================================================

void KTruss::reset() {
    cudaMemset(hd_data().counter,0, sizeof(int));
    hd_data().num_edges_remaining      = originalNE;
    hd_data().full_triangle_iterations = 0;

    resetEdgeArray();
    resetVertexArray();
}

void KTruss::resetVertexArray() {
    gpu::memsetZero(hd_data().triangles_per_vertex, originalNV);
}

void KTruss::resetEdgeArray() {
    gpu::memsetZero(hd_data().triangles_per_edge, originalNE);
}

void KTruss::release() {
    gpu::free(hd_data().is_active);
    gpu::free(hd_data().offset_array);
    gpu::free(hd_data().triangles_per_edge);
    gpu::free(hd_data().triangles_per_vertex);
    gpu::free(hd_data().counter);
    gpu::free(hd_data().active_vertices);

    hd_data().is_active            = nullptr;
    hd_data().offset_array         = nullptr;
    hd_data().triangles_per_edge   = nullptr;
    hd_data().triangles_per_vertex = nullptr;
    hd_data().counter              = nullptr;
    hd_data().active_vertices      = nullptr;

}

//==============================================================================

void KTruss::run() {
    hd_data().max_K = 3;
    int  iterations = 0;

    while (true) {
        bool need_stop = false;

        bool      more = findTrussOfK(need_stop);

        if (hd_data().num_edges_remaining <= 0) {
            hd_data().max_K--;
            break;
        }
        hd_data().max_K++;

        iterations++;
    }
}

void KTruss::runForK(int max_K) {
    hd_data().max_K = max_K;

    bool exit_on_first_iteration;
    findTrussOfK(exit_on_first_iteration);
}

bool KTruss::findTrussOfK(bool& stop) {
    forAllVertices(hornet, Init { hd_data });
    resetEdgeArray();
    resetVertexArray();
 
    cudaMemset(hd_data().counter,0, sizeof(int));

    int h_active_vertices = originalNV;

    stop = true;

    while (h_active_vertices > 0) {

        hd_data().full_triangle_iterations++;

        kTrussOneIteration(hornet, hd_data().triangles_per_vertex,
                           hd_data().tsp, hd_data().nbl,
                           hd_data().shifter,
                           hd_data().blocks, hd_data().sps,
                           hd_data);

        forAllVertices(hornet, FindUnderK { hd_data });

        int h_counter;
        cudaMemcpy(&h_counter,hd_data().counter, sizeof(int),cudaMemcpyDeviceToHost);

        if (h_counter != 0) {
              UpdatePtr ptr(h_counter, hd_data().src, hd_data().dst);
              Update batch_update(ptr);
              hornet.erase(batch_update);
        }
        else{
            return false;
        }

        hd_data().num_edges_remaining -= h_counter;

        // Resetting the number of active vertices before check
        cudaMemset(hd_data().active_vertices,0, sizeof(int));
        forAllVertices(hornet, CountActive { hd_data });

        sortHornet();


        // Getting the number of active vertices
        cudaMemcpy(&h_active_vertices, hd_data().active_vertices,sizeof(int),cudaMemcpyDeviceToHost);

        resetEdgeArray();
        resetVertexArray();

        cudaMemset(hd_data().counter,0, sizeof(int));
        stop = false;

    }
    return true;
}


} // hornet_alg namespace
