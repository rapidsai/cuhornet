/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

namespace hornets_nest {

struct Init {
    HostDeviceVar<KTrussData> kt;

    OPERATOR(Vertex& vertex) {
        vert_t           src = vertex.id();
        kt().is_active[src] = 1;
    }
};


struct FindUnderK {
    HostDeviceVar<KTrussData> kt;

    OPERATOR(Vertex& vertex) {
        vert_t src = vertex.id();

        if (kt().is_active[src] == 0)
            return;
        if (vertex.degree() == 0) {
            kt().is_active[src] = 0;
            return;
        }
        for (vert_t adj = 0; adj < vertex.degree(); adj++) {
            int   pos = kt().offset_array[src] + adj;
            if (kt().triangles_per_edge[pos] < (kt().max_K - 2)) {
                int       spot = atomicAdd((kt().counter), 1);
                kt().src[spot] = src;
                vert_t dest = vertex.neighbor_ptr()[adj];
                kt().dst[spot] = dest;
            }
        }
    }
};

struct getVertexSizes {
    int* sizes;

    OPERATOR(Vertex& vertex) {
        vert_t src = vertex.id();
        sizes[src] = vertex.degree();
    }
};


struct SimpleBubbleSort {

    OPERATOR(Vertex& vertex) {
        vert_t src = vertex.id();

        degree_t size = vertex.degree();
        if(size<=1)
            return;
        for (vert_t i = 0; i < (size-1); i++) {
            vert_t min_idx=i;

            for(vert_t j=i+1; j<(size); j++){
                if(vertex.neighbor_ptr()[j]<vertex.neighbor_ptr()[min_idx])
                    min_idx=j;
            }
            vert_t temp = vertex.neighbor_ptr()[i];
            vertex.neighbor_ptr()[i] = vertex.neighbor_ptr()[min_idx];
            vertex.neighbor_ptr()[min_idx] = temp;
        }
 
    }
};


struct CountActive {
    HostDeviceVar<KTrussData> kt;

    OPERATOR(Vertex& vertex) {
        vert_t src = vertex.id();

        if (vertex.degree() == 0 && !kt().is_active[src])
            kt().is_active[src] = 0;
        else
            atomicAdd((kt().active_vertices), 1);
    }
};

} // namespace hornets_nest
