 
/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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


/*
Details of Logarthmic Radix Binning (LRB for short) can be found in additional detail in

[1] Fox, James, Alok Tripathy, and Oded Green
    "Improving Scheduling for Irregular Applications with Logarithmic Radix Binning."
    IEEE High Performance Extreme Computing Conference (HPEC), 2019

[2] Green, Oded, James Fox, Alex Watkins, Alok Tripathy, Kasimir Gabert, Euna Kim,
    Xiaojing An, Kumar Aatish, and David A. Bader.
    "Logarithmic radix binning and vectorized triangle counting."
    IEEE High Performance extreme Computing Conference (HPEC), 2018.

This implementation of LRB follows the details in [1]. More details of LRB can be found in [2].
*/

#pragma once

#include <cuda_runtime.h>
#include "BasicTypes.hpp"

namespace hornets_nest {
/**
 * @brief The namespace provides all load balancing methods to traverse vertices
 */
namespace load_balancing {

/**
 * @brief The class implements the LogarthimRadixBinning32 load balancing
 */
class LogarthimRadixBinning32 {
public:
    /**
     * @brief Default costructor
     * @param[in] hornet Hornet instance
     */
    template<typename HornetClass>
    explicit LogarthimRadixBinning32(HornetClass& hornet,
                          const float work_factor = 2.0f) noexcept;

    /**
     * @brief Decostructor
     */
    ~LogarthimRadixBinning32() noexcept;

    /**
     * @brief Traverse the edges in a vertex queue (C++11-Style API)
     * @tparam Operator function to apply at each edge
     * @param[in] queue input vertex queue
     * @param[in] op struct/lambda expression that implements the operator
     * @remark    all algorithm-dependent data must be capture by `op`
     * @remark    the Operator typename must implement the method
     *            `void operator()(Vertex, Edge)` or the lambda expression
     *            `[=](Vertex, Edge){}`
     */
     template<typename HornetClass, typename Operator, typename vid_t>
     void apply(HornetClass& hornet,
                const vid_t *      d_input,
                int                num_vertices,
                const Operator&    op) const noexcept;

    template<typename HornetClass, typename Operator>
    void apply(HornetClass& hornet, const Operator& op) const noexcept;

private:
    static const unsigned BLOCK_SIZE = 128;

    mutable xlib::CubExclusiveSum<int> prefixsum;

    int32_t*      d_lrbRelabled;
    int32_t*      d_bins;
    int32_t*      d_binsPrefix;

    cudaEvent_t syncher;
    const int STREAMS = 12;
    cudaStream_t* streams;

};

} // namespace load_balancing
} // namespace hornets_nest

#include "LoadBalancing/lrb_lb.i.cuh"
