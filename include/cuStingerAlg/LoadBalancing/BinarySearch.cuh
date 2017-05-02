/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "Core/cuStingerTypes.cuh"

/**
 * @brief
 */
namespace load_balacing {

class BinarySearch {
public:
    explicit BinarySearch(const cu_stinger::eoff_t* csr_offsets,
                          size_t num_vertices) noexcept;
    explicit BinarySearch(const cu_stinger::eoff_t* csr_offsets,
                          size_t num_vertices, int max_allocated_items)
                          noexcept;
    ~BinarySearch() noexcept;

    template<void (*Operator)(cu_stinger::Vertex, cu_stinger::Edge, void*)>
    void traverse_edges(const cu_stinger::vid_t* d_input, int num_vertices,
                        void* optional_field) noexcept;

    template<typename Operator>
    void traverse_edges(const cu_stinger::vid_t* d_input, int num_vertices,
                        Operator op) noexcept;

private:
    static const int         BLOCK_SIZE = 256;
    static const bool CHECK_CUDA_ERROR1 = 1;
    int* _d_work    { nullptr };
    int* _d_degrees { nullptr };
};

} // namespace load_balacing

#include "cuStingerAlg/LoadBalancing/BinarySearch.i.cuh"