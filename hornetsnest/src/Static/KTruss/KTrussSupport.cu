// #include "Static/KTruss/KTruss.cuh"

namespace hornets_nest {

__device__ __forceinline__
void initialize(vert_t diag_id,
                vert_t u_len,
                vert_t v_len,
                vert_t* __restrict__ u_min,
                vert_t* __restrict__ u_max,
                vert_t* __restrict__ v_min,
                vert_t* __restrict__ v_max,
                int*   __restrict__ found) {
    if (diag_id == 0) {
        *u_min = *u_max = *v_min = *v_max = 0;
        *found = 1;
    }
    else if (diag_id < u_len) {
        *u_min = 0;
        *u_max = diag_id;
        *v_max = diag_id;
        *v_min = 0;
    }
    else if (diag_id < v_len) {
        *u_min = 0;
        *u_max = u_len;
        *v_max = diag_id;
        *v_min = diag_id - u_len;
    }
    else {
        *u_min = diag_id - v_len;
        *u_max = u_len;
        *v_min = diag_id - u_len;
        *v_max = v_len;
    }
}

__device__ __forceinline__
void workPerThread(vert_t uLength,
                   vert_t vLength,
                   int threadsPerIntersection,
                   int threadId,
                   int* __restrict__ outWorkPerThread,
                   int* __restrict__ outDiagonalId) {
  int      totalWork = uLength + vLength;
  int  remainderWork = totalWork % threadsPerIntersection;
  int  workPerThread = totalWork / threadsPerIntersection;

  int longDiagonals  = threadId > remainderWork ? remainderWork : threadId;
  int shortDiagonals = threadId > remainderWork ? threadId - remainderWork : 0;

  *outDiagonalId     = (workPerThread + 1) * longDiagonals +
                        workPerThread * shortDiagonals;
  *outWorkPerThread  = workPerThread + (threadId < remainderWork);
}

__device__ __forceinline__
void bSearch(unsigned found,
             vert_t    diagonalId,
             const vert_t*  __restrict__ uNodes,
             const vert_t*  __restrict__ vNodes,
             const vert_t*  __restrict__ uLength,
             vert_t* __restrict__ outUMin,
             vert_t* __restrict__ outUMax,
             vert_t* __restrict__ outVMin,
             vert_t* __restrict__ outVMax,
             vert_t* __restrict__ outUCurr,
             vert_t* __restrict__ outVCurr) {
    vert_t length;

    while (!found) {
        *outUCurr = (*outUMin + *outUMax) >> 1;
        *outVCurr = diagonalId - *outUCurr;
        if (*outVCurr >= *outVMax) {
            length = *outUMax - *outUMin;
            if (length == 1) {
                found = 1;
                continue;
            }
        }

        unsigned comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
        unsigned comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
        if (comp1 && !comp2)
            found = 1;
        else if (comp1) {
          *outVMin = *outVCurr;
          *outUMax = *outUCurr;
        }
        else {
          *outVMax = *outVCurr;
          *outUMin = *outUCurr;
        }
      }

    if (*outVCurr >= *outVMax && length == 1 && *outVCurr > 0 &&
        *outUCurr > 0 && *outUCurr < *uLength - 1) {
        unsigned comp1 = uNodes[*outUCurr] > vNodes[*outVCurr - 1];
        unsigned comp2 = uNodes[*outUCurr - 1] > vNodes[*outVCurr];
        if (!comp1 && !comp2) {
            (*outUCurr)++;
            (*outVCurr)--;
        }
    }
}

__device__ __forceinline__
int fixStartPoint(vert_t uLength, vert_t vLength,
                  vert_t* __restrict__ uCurr,
                  vert_t* __restrict__ vCurr,
                  const vert_t* __restrict__ uNodes,
                  const vert_t* __restrict__ vNodes) {

    unsigned uBigger = (*uCurr > 0) && (*vCurr < vLength) &&
                       (uNodes[*uCurr - 1] == vNodes[*vCurr]);
    unsigned vBigger = (*vCurr > 0) && (*uCurr < uLength) &&
                       (vNodes[*vCurr - 1] == uNodes[*uCurr]);
    *uCurr += vBigger;
    *vCurr += uBigger;
    return uBigger + vBigger;
}

/*
__device__ __forceinline__
vert_t* binSearch(vert_t *a, vertexId_t x, vert_t n) {
    vert_t min = 0, max = n, acurr, curr;// = (min+max)/2
    do {
        curr  = (min + max) / 2;
        acurr = a[curr];
        min   = (x > acurr) ? curr : min;
        max   = (x < acurr) ? curr : max;
    } while (x != acurr || min != max);
    return a + curr;
}*/

/*
__device__ __forceinline__
int findIndexOfVertex(HornetGraph* hornet, vert_t src, vert_t dst__) {
    vert_t   srcLen = hornet->dVD->used[src];
    vert_t* adj_src = hornet->dVD->adj[src]->dst;

    for (vert_t adj = 0; adj < srcLen; adj++) {
        vert_t dst = adj_src[adj];
        if (dst == dst__)
            return adj;
    }
#if !defined(NDEBUG)
    printf("This should never happpen\n");
#endif
    return -1;
}*/

__device__ __forceinline__
void indexBinarySearch(vert_t* data, vert_t arrLen, vert_t key, int& pos) {
    int  low = 0;
    int high = arrLen - 1;
    while (high >= low) {
        int middle = (low + high) / 2;
        if (data[middle] == key) {
             pos = middle;
             return;
        }
        if (data[middle] < key)
            low = middle + 1;
        if (data[middle] > key)
            high = middle - 1;
    }
}

// template<typename Vertex>
// __device__ __forceinline__
// void findIndexOfTwoVerticesBinary(const Vertex& vertex,
//                                   vert_t v1, vert_t v2,
//                                   int &pos_v1, int &pos_v2) {
//     //vert_t* adj_src = hornet->dVD->adj[src]->dst;
//     //vert_t   srcLen = hornet->dVD->used[src];
//     vert_t   srcLen = vertex.degree();
//     vert_t* adj_src = vertex.neighbor_ptr();

//     pos_v1 = -1;
//     pos_v2 = -1;

//     indexBinarySearch(adj_src, srcLen, v1, pos_v1);
//     indexBinarySearch(adj_src, srcLen, v2, pos_v2);
// }

// template<typename Vertex>
// __device__ __forceinline__
// void findIndexOfTwoVertices(const Vertex& vertex, vert_t v1, vert_t v2,
//                             int &pos_v1, int &pos_v2) {
//     //vert_t   srcLen = hornet->dVD->used[src];
//     //vert_t* adj_src = hornet->dVD->adj[src]->dst;
//     vert_t   srcLen = vertex.degree();
//     vert_t* adj_src = vertex.neighbor_ptr();

//     pos_v1 = -1;
//     pos_v2 = -1;
//     for(vert_t adj = 0; adj < srcLen; adj += 1) {
//         vert_t dst = adj_src[adj];
//         if (dst == v1)
//             pos_v1 = adj;
//         if (dst == v2)
//             pos_v2 = adj;
//         if (pos_v1 != -1 && pos_v2 != -1)
//             return;
//     }
// #if !defined(NDEBUG)
//     printf("This should never happpen\n");
// #endif
// }

template<bool uMasked, bool vMasked, bool subtract, bool upd3rdV,
         typename HornetDevice>
__device__ __forceinline__
void intersectCount(HornetDevice& hornet,
                    vert_t uLength, vert_t vLength,
                    const vert_t* __restrict__ uNodes,
                    const vert_t* __restrict__ vNodes,
                    vert_t*       __restrict__ uCurr,
                    vert_t*       __restrict__ vCurr,
                    int*         __restrict__ workIndex,
                    const int*   __restrict__ workPerThread,
                    int*         __restrict__ triangles,
                    int found,
                    const triangle_t*  __restrict__ output_triangles,
                    const vert_t*  __restrict__ uMask,
                    const vert_t*  __restrict__ vMask,
                    triangle_t multiplier,
                    vert_t src, vert_t dest,
                    vert_t u, vert_t v) {

    if (*uCurr < uLength && *vCurr < vLength) {
        int comp;
        int vmask;
        int umask;
        while (*workIndex < *workPerThread) {
            // vmask = vMasked ? vMask[*vCurr] : 0;
            // umask = uMasked ? uMask[*uCurr] : 0;
            vmask=umask=0;
            comp  = uNodes[*uCurr] - vNodes[*vCurr];

            *triangles += (comp == 0);

            *uCurr     += (comp <= 0 && !vmask) || umask;
            *vCurr     += (comp >= 0 && !umask) || vmask;
            *workIndex += (comp == 0 && !umask && !vmask) + 1;

            if (*vCurr >= vLength || *uCurr >= uLength)
                break;
            // comp  = uNodes[*uCurr] - vNodes[*vCurr];

            // *triangles += (comp == 0);
            // *uCurr     += (comp <= 0) ;
            // *vCurr     += (comp >= 0);
            // *workIndex += (comp == 0) + 1;

            // if (*vCurr >= vLength || *uCurr >= uLength)
            //     break;


        }
        *triangles -= ((comp == 0) && (*workIndex > *workPerThread) && found);
    }
}

// u_len < v_len
template <bool uMasked, bool vMasked, bool subtract, bool upd3rdV,
          typename HornetDevice>
__device__ __forceinline__
triangle_t count_triangles(HornetDevice& hornet,
                           vert_t u,
                           const vert_t* __restrict__ u_nodes,
                           vert_t u_len,
                           vert_t v,
                           const vert_t* __restrict__ v_nodes,
                           vert_t v_len,
                           int   threads_per_block,
                           volatile vert_t* __restrict__ firstFound,
                           int    tId,
                           const triangle_t* __restrict__ output_triangles,
                           const vert_t*      __restrict__ uMask,
                           const vert_t*      __restrict__ vMask,
                           triangle_t multiplier,
                           vert_t      src,
                           vert_t      dest) {

    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements to
    //Tersect - this number will be off by 1.
    int work_per_thread, diag_id;
    workPerThread(u_len, v_len, threads_per_block, tId,
                  &work_per_thread, &diag_id);
    triangle_t triangles = 0;
    int       work_index = 0;
    int            found = 0;
    vert_t u_min, u_max, v_min, v_max, u_curr, v_curr;

    firstFound[tId] = 0;

    if (work_per_thread > 0) {
        // For the binary search, we are figuring out the initial poT of search.
        initialize(diag_id, u_len, v_len, &u_min, &u_max,
                   &v_min, &v_max, &found);
        u_curr = 0;
        v_curr = 0;

        bSearch(found, diag_id, u_nodes, v_nodes, &u_len, &u_min, &u_max,
                &v_min, &v_max, &u_curr, &v_curr);

        int sum = fixStartPoint(u_len, v_len, &u_curr, &v_curr,
                                u_nodes, v_nodes);
        work_index += sum;
        if (tId > 0)
           firstFound[tId - 1] = sum;
        triangles += sum;
        intersectCount<uMasked, vMasked, subtract, upd3rdV>
            (hornet, u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
            &work_index, &work_per_thread, &triangles, firstFound[tId],
            output_triangles, uMask, vMask, multiplier, src, dest, u, v);
    }
    return triangles;
}

__device__ __forceinline__
void workPerBlock(vert_t numVertices,
                  vert_t* __restrict__ outMpStart,
                  vert_t* __restrict__ outMpEnd,
                  int blockSize) {
    vert_t       verticesPerMp = numVertices / gridDim.x;
    vert_t     remainderBlocks = numVertices % gridDim.x;
    vert_t   extraVertexBlocks = (blockIdx.x > remainderBlocks) ? remainderBlocks
                                                               : blockIdx.x;
    vert_t regularVertexBlocks = (blockIdx.x > remainderBlocks) ?
                                    blockIdx.x - remainderBlocks : 0;

    vert_t mpStart = (verticesPerMp + 1) * extraVertexBlocks +
                     verticesPerMp * regularVertexBlocks;
    *outMpStart   = mpStart;
    *outMpEnd     = mpStart + verticesPerMp + (blockIdx.x < remainderBlocks);
}

//==============================================================================
//==============================================================================

template<typename HornetDevice>
__global__
void devicecuStingerKTruss(HornetDevice hornet,
                           const triangle_t* __restrict__ output_triangles,
                           int threads_per_block,
                           int number_blocks,
                           int shifter,
                           HostDeviceVar<KTrussData> hd_data) {
    vert_t nv = hornet.nV();

    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements
    //to intersect - this number will be off by no more than one.
    int tx = threadIdx.x;
    vert_t this_mp_start, this_mp_stop;

    const int blockSize = blockDim.x;
    workPerBlock(nv, &this_mp_start, &this_mp_stop, blockSize);

    //__shared__ triangle_t s_triangles[1024];
    __shared__ vert_t      firstFound[1024];

    vert_t     adj_offset = tx >> shifter;
    vert_t* firstFoundPos = firstFound + (adj_offset << shifter);
    for (vert_t src = this_mp_start; src < this_mp_stop; src++) {
        //vert_t      srcLen = hornet->dVD->getUsed()[src];
        auto vertex = hornet.vertex(src);
        vert_t  srcLen = vertex.degree();

        // triangle_t tCount = 0;
        for(int k = adj_offset; k < srcLen; k += number_blocks) {
            //vert_t  dest = hornet->dVD->getAdj()[src]->dst[k];
            vert_t dest = vertex.edge(k).dst_id();
            //int destLen = hornet->dVD->getUsed()[dest];
            int destLen = hornet.vertex(dest).degree();

            // if (dest < src) //opt
            //     continue;   //opt

            bool avoidCalc = (src == dest) || (destLen < 2) || (srcLen < 2);
            if (avoidCalc)
                continue;

            bool sourceSmaller = srcLen < destLen;
            vert_t        small = sourceSmaller ? src : dest;
            vert_t        large = sourceSmaller ? dest : src;
            vert_t    small_len = sourceSmaller ? srcLen : destLen;
            vert_t    large_len = sourceSmaller ? destLen : srcLen;

            //const vert_t* small_ptr = hornet->dVD->getAdj()[small]->dst;
            //const vert_t* large_ptr = hornet->dVD->getAdj()[large]->dst;
            const vert_t* small_ptr = hornet.vertex(small).neighbor_ptr();
            const vert_t* large_ptr = hornet.vertex(large).neighbor_ptr();

            // triangle_t triFound = count_triangles<false,false,false,true>
            triangle_t triFound = count_triangles<false, false, false, false>
                (hornet, small, small_ptr, small_len, large, large_ptr,
                 large_len, threads_per_block, firstFoundPos,
                 tx % threads_per_block, output_triangles,
                 nullptr, nullptr, 1, src, dest);
            // tCount += triFound;
            int pos = hd_data().offset_array[src] + k;
            atomicAdd(hd_data().triangles_per_edge + pos, triFound);
            // pos = -1; //opt
            // //indexBinarySearch(hornet->dVD->getAdj()[dest]->dst
            // //                  destLen, src,pos);
            // auto dest_ptr = hornet.vertex(dest).neighbor_ptr();
            // indexBinarySearch(dest_ptr, destLen, src, pos);

            // pos = hd_data().offset_array[dest] + pos;
            // atomicAdd(hd_data().triangles_per_edge + pos, triFound);
        }
    //    s_triangles[tx] = tCount;
    //    blockReduce(&output_triangles[src],s_triangles,blockSize);
    }
}

//==============================================================================

void kTrussOneIteration(HornetGraph& hornet,
                        const triangle_t* __restrict__ output_triangles,
                        int threads_per_block,
                        int number_blocks,
                        int shifter,
                        int thread_blocks,
                        int blockdim,
                        HostDeviceVar<KTrussData>& hd_data) {

    //devicecuStingerKTruss <<< thread_blocks, blockdim >>>
    //    (hornet.devicePtr(), output_triangles, threads_per_block,
    //     number_blocks, shifter, devData);
    devicecuStingerKTruss <<< thread_blocks, blockdim >>>
        (hornet.device(), output_triangles, threads_per_block,
         number_blocks, shifter, hd_data);

}
/*
//==============================================================================
//==============================================================================
template<typename HornetDevice>
__global__
void devicecuStingerNewTriangles(HornetDevice hornet,
                                 gpu::BatchUpdate batch_update,
                                 const triangle_t* __restrict__ output_triangles,
                                 int threads_per_block,
                                 int number_blocks,
                                 int shifter,
                                 bool deletion) {
    //vert_t batchSize = *(batch_update->getBatchSize());
    vert_t batchSize = batch_update.size();
    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements to
    //intersect - this number will be off by no more than one.
    int tx = threadIdx.x;
    vert_t this_mp_start, this_mp_stop;

    //vert_t* d_ind = batch_update->getDst();
    //vert_t* d_seg = batch_update->getSrc();
    vert_t* d_ind = batch_update.dst_ptr();
    vert_t* d_seg = batch_update.src_ptr();

    workPerBlock(batchSize, &this_mp_start, &this_mp_stop, blockDim.x);

    __shared__ vert_t firstFound[1024];

    vert_t     adj_offset = tx >> shifter;
    vert_t* firstFoundPos = firstFound + (adj_offset << shifter);
    for (vert_t edge = this_mp_start + adj_offset; edge < this_mp_stop;
         edge += number_blocks){
        //if (batch_update->getIndDuplicate()[edge] == 1) // this means it's a duplicate edge
        //    continue;

        vert_t src  = d_seg[edge];
        vert_t dest = d_ind[edge];

        if (src < dest)
            continue;

        vert_t srcLen  = hornet.vertex(src).degree();
        vert_t destLen = hornet.vertex(dest).degree();
        //vert_t srcLen  = hornet->dVD->getUsed()[src];
        //vert_t destLen = hornet->dVD->getUsed()[dest];

        bool avoidCalc = (src == dest) || (destLen == 0) || (srcLen == 0);
        if (avoidCalc)
            continue;

        bool sourceSmaller = srcLen < destLen;
        vert_t        small = sourceSmaller ? src : dest;
        vert_t        large = sourceSmaller ? dest : src;
        vert_t    small_len = sourceSmaller ? srcLen : destLen;
        vert_t    large_len = sourceSmaller ? destLen : srcLen;

        //const vert_t* small_ptr = hornet->dVD->getAdj()[small]->dst;
        //const vert_t* large_ptr = hornet->dVD->getAdj()[large]->dst;
        const vert_t* small_ptr = hornet.vertex(small).neighbor_ptr();
        const vert_t* large_ptr = hornet.vertex(large).neighbor_ptr();

        triangle_t tCount = count_triangles<false, false, true, true>(
                                hornet, small, small_ptr, small_len,
                                large, large_ptr, large_len,
                                threads_per_block, firstFoundPos,
                                tx % threads_per_block, output_triangles,
                                nullptr, nullptr, 2, src, dest);
        __syncthreads();
    }
}
*/
//==============================================================================
/*
template <bool uMasked, bool vMasked, bool subtract, bool upd3rdV,
          typename HornetDevice>
__device__ __forceinline__
void intersectCountAsymmetric(HornetDevice& hornet,
                              vert_t uLength, vert_t vLength,
                              const vert_t* __restrict__ uNodes,
                              const vert_t* __restrict__ vNodes,
                              vert_t* __restrict__ uCurr,
                              vert_t* __restrict__ vCurr,
                              int*   __restrict__ workIndex,
                              const int*   __restrict__ workPerThread,
                              int*   __restrict__ triangles,
                              int found,
                              triangle_t* __restrict__ output_triangles,
                              const vert_t*      __restrict__ uMask,
                              const vert_t*      __restrict__ vMask,
                              triangle_t multiplier,
                              vert_t src, vert_t dest,
                              vert_t u, vert_t v) {

    // if(u==0)
    //   printf("|u|=%d\n",uLength);
    // if(v==0)
    //   printf("|v|=%d\n",vLength);

    // printf("%d %d\n",u,v);

    if (*uCurr < uLength && *vCurr < vLength) {
        int comp, vmask, umask;
        while (*workIndex < *workPerThread) {
            // vmask = vMasked ? vMask[*vCurr] : 0;
            // umask = uMasked ? uMask[*uCurr] : 0;
            umask=vmask=0;
            comp  = uNodes[*uCurr] - vNodes[*vCurr];

            // *triangles += (comp == 0 && !umask && !vmask);
            *triangles += (comp == 0);

            // if (upd3rdV && comp == 0 && !umask && !vmask) {
            if (upd3rdV && comp == 0) {
                if (subtract) {
                    // atomicSub(output_triangles + uNodes[*uCurr], multiplier);
                    // if(blockIdx.x<=10)
                    //   printf("!!! %d %d", u,v);

                    // Ktruss
                    //vert_t common = uNodes[*uCurr];

                    if (dest == u) {
                        auto w_ptr = hornet.vertex(dest).edge_weight_ptr();
                        atomicSub(w_ptr + *uCurr, 1);
                        //atomicSub(hornet->dVD->adj[dest]->ew + *uCurr, 1);
                    }
                    else {
                        auto w_ptr = hornet.vertex(dest).edge_weight_ptr();
                        atomicSub(w_ptr + *vCurr, 1);
                        //atomicSub(hornet->dVD->adj[dest]->ew + *vCurr, 1);
                    }
                }
            }
            *uCurr     += (comp <= 0 && !vmask) || umask;
            *vCurr     += (comp >= 0 && !umask) || vmask;
            *workIndex += (comp == 0 && !umask && !vmask) + 1;

            if (*vCurr == vLength || *uCurr == uLength)
                break;
        }
        *triangles -= ((comp == 0) && (*workIndex > *workPerThread) && (found));
    }
}
*/
//==============================================================================
//==============================================================================
/*
// u_len < v_len
template <bool uMasked, bool vMasked, bool subtract, bool upd3rdV,
          typename HornetDevice>
__device__ __forceinline__
triangle_t count_trianglesAsymmetric(
                                 HornetDevice& hornet,
                                 vert_t u,
                                 const vert_t* __restrict__ u_nodes,
                                 vert_t u_len,
                                 vert_t v,
                                 const vert_t* __restrict__ v_nodes,
                                 vert_t v_len,
                                 int threads_per_block,
                                 volatile vert_t* __restrict__ firstFound,
                                 int tId,
                                 triangle_t* __restrict__ output_triangles,
                                 const vert_t* __restrict__ uMask,
                                 const vert_t* __restrict__ vMask,
                                 triangle_t multiplier,
                                 vert_t src, vert_t dest) {
    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements to
    // Tersect - this number will be off by 1.
    int work_per_thread, diag_id;
    workPerThread(u_len, v_len, threads_per_block, tId,
                  &work_per_thread, &diag_id);
    triangle_t triangles = 0;
    int       work_index = 0;
    int            found = 0;
    vert_t u_min, u_max, v_min, v_max, u_curr, v_curr;

    firstFound[tId] = 0;

    if (work_per_thread > 0) {
        // For the binary search, we are figuring out the initial poT of search.
        initialize(diag_id, u_len, v_len, &u_min, &u_max,
                   &v_min, &v_max, &found);
        u_curr = 0;
        v_curr = 0;

        bSearch(found, diag_id, u_nodes, v_nodes, &u_len, &u_min, &u_max,
                &v_min, &v_max, &u_curr, &v_curr);

        int sum = fixStartPoint(u_len, v_len, &u_curr, &v_curr,
                                u_nodes, v_nodes);
        work_index += sum;
        if (tId > 0)
            firstFound[tId - 1] = sum;
        triangles += sum;
        intersectCountAsymmetric<uMasked, vMasked, subtract, upd3rdV>
            (hornet, u_len, v_len, u_nodes, v_nodes, &u_curr, &v_curr,
             &work_index, &work_per_thread, &triangles, firstFound[tId],
             output_triangles, uMask, vMask, multiplier, src, dest, u, v);
    }
    return triangles;
}
*/
//==============================================================================
//==============================================================================
/*
__device__ int d_value[32];

template<typename HornetDevice>
__global__
void deviceBUTwoCUOneTriangles(HornetDevice hornet,
                               gpu::BatchUpdate batch_update,
                               triangle_t* __restrict__ output_triangles,
                               int  threads_per_block,
                               int  number_blocks,
                               int  shifter,
                               bool deletion) {
    //vert_t batchsize = *(batch_update->getBatchSize());
    vert_t batchsize = batch_update.size();

    // Partitioning the work to the multiple thread of a single GPU processor.
    //The threads should get a near equal number of the elements to
    //intersect - this number will be off by no more than one.
    int tx = threadIdx.x;
    vert_t this_mp_start, this_mp_stop;

    //vert_t* d_off = batch_update->getOffsets();
    const vert_t* d_off = batch_update.csr_wide_offsets_ptr();

    //vert_t* d_ind = batch_update->getDst();
    //vert_t* d_seg = batch_update->getSrc();
    vert_t* d_ind = batch_update.dst_ptr();
    vert_t* d_seg = batch_update.src_ptr();

    int blockSize = blockDim.x;
    workPerBlock(batchsize, &this_mp_start, &this_mp_stop, blockSize);

    __shared__ vert_t firstFound[1024];

    vert_t     adj_offset = tx >> shifter;
    vert_t* firstFoundPos = firstFound + (adj_offset << shifter);
    for (vert_t edge = this_mp_start + adj_offset; edge < this_mp_stop;
            edge += number_blocks) {
        //if (batch_update->getIndDuplicate()[edge]) // this means it's a duplicate edge
        //    continue;

        assert(edge < batch_update.size());

        vert_t src  = batch_update.src(edge);
        vert_t dest = batch_update.dst(edge);

        vert_t  srcLen = d_off[src + 1] - d_off[src];
        vert_t destLen = hornet.vertex(dest).degree();

        bool avoidCalc = src == dest || srcLen == 0;
        if (avoidCalc)
            continue;

        const vert_t*      src_ptr = d_ind + d_off[src];
        //const vert_t* src_mask_ptr = batch_update->getIndDuplicate() + d_off[src];//???
        const vert_t* src_mask_ptr = nullptr;
        //const vert_t*      dst_ptr = hornet->dVD->getAdj()[dest]->dst;
        const vert_t*      dst_ptr = hornet.vertex(dest).neighbor_ptr();

        assert(d_off[src] < batch_update.size());

        bool sourceSmaller = srcLen < destLen;
        vert_t        small = sourceSmaller ? src : dest;
        vert_t        large = sourceSmaller ? dest : src;
        vert_t    small_len = sourceSmaller ? srcLen : destLen;
        vert_t    large_len = sourceSmaller ? destLen : srcLen;

        const vert_t*      small_ptr = sourceSmaller ? src_ptr : dst_ptr;
        const vert_t* small_mask_ptr = sourceSmaller ? src_mask_ptr : nullptr;
        const vert_t*      large_ptr = sourceSmaller ? dst_ptr : src_ptr;
        const vert_t* large_mask_ptr = sourceSmaller ? nullptr : src_mask_ptr;

        // triangle_t tCount=0;
        triangle_t tCount = sourceSmaller ?
                            count_trianglesAsymmetric<false, false, true, true>
                                (hornet, small, small_ptr, small_len,
                                  large, large_ptr, large_len,
                                 threads_per_block, firstFoundPos,
                                 tx % threads_per_block, output_triangles,
                                   small_mask_ptr, large_mask_ptr, 1,src,dest) :
                            count_trianglesAsymmetric<false, false, true, true>
                                (hornet, small, small_ptr, small_len,
                                 large, large_ptr, large_len,
                                   threads_per_block, firstFoundPos,
                                 tx % threads_per_block, output_triangles,
                                 small_mask_ptr, large_mask_ptr, 1, src, dest);

        // atomicSub(output_triangles + src, tCount * 1);
        // atomicSub(output_triangles + dest, tCount * 1);
        __syncthreads();
    }
}
*/
/*
void callDeviceDifferenceTriangles(
                                const HornetGraph& hornet,
                                const gpu::BatchUpdate& batch_update,
                                triangle_t* __restrict__ output_triangles,
                                int  threads_per_intersection,
                                int  num_intersec_perblock,
                                int  shifter,
                                int  thread_blocks,
                                int  blockdim,
                                bool deletion) {
    dim3 numBlocks(1, 1);
    //vert_t batchsize = *(batch_update.getHostBUD()->getBatchSize());
    //vert_t        nv = *(batch_update.getHostBUD()->getNumVertices());
    vert_t batchsize = batch_update.size();

    //vert_t        nv = *(batch_update.getHostBUD()->getNumVertices());
    vert_t        nv = hornet.nV();

    numBlocks.x = ceil( (float) nv / (float) blockdim );
    //vert_t* redCU;
    //vert_t* redBU;

    numBlocks.x = ceil( (float) (batchsize * threads_per_intersection) /
                        (float) blockdim );

    // cout << "The block dim is " << blockdim << " and the number of blocks is"
    //<< numBlocks.x << endl;
    // Calculate all new traingles regardless of repetition
    devicecuStingerNewTriangles <<< numBlocks, blockdim >>>
        (hornet.device_side(), batch_update,
         output_triangles, threads_per_intersection, num_intersec_perblock,
         shifter, deletion);

    // Calculate triangles formed by ALL new edges
        // deviceBUThreeTriangles<<<numBlocks,blockdim>>>(hornet.devicePtr(),
    //    batch_update.getDeviceBUD()->devicePtr(), output_triangles,
    //threads_per_intersection,num_intersec_perblock,shifter,deletion,redBU);

    // Calculate triangles formed by two new edges
    deviceBUTwoCUOneTriangles <<< numBlocks, blockdim >>>
        (hornet.device_side(), batch_update,
        output_triangles, threads_per_intersection, num_intersec_perblock,
        shifter, deletion);
}
*/
} // namespace hornets_nest
