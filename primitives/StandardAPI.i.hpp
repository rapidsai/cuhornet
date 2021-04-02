/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
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
 */
#pragma once

#include <Device/Util/Algorithm.cuh>
#include <Device/Util/PrintExt.cuh>
#include <Device/Util/SafeCudaAPI.cuh>
#include <Device/Util/SafeCudaAPIAsync.cuh>
#include <Device/Primitives/CubWrapper.cuh>
#include <omp.h>
#include <cstring>
#include <rmm/mr/device/per_device_resource.hpp>

namespace hornets_nest {
namespace gpu {

template<typename T>
void allocate(T*& pointer, size_t num_items) {
    pointer = static_cast<T*>(
        rmm::mr::get_current_device_resource()->allocate(sizeof(T)*num_items,
          rmm::cuda_stream_default));
}

template<typename T>
typename std::enable_if<std::is_pointer<T>::value>::type
free(T& pointer, size_t num_items) {
    rmm::mr::get_current_device_resource()->deallocate(static_cast<void*>(pointer), sizeof(T)*num_items,
        rmm::cuda_stream_default);
}

template<typename T>
void copyToDevice(const T* device_input, size_t num_items, T* device_output) {
    cuMemcpyDevToDev(device_input, num_items, device_output);
}

template<typename T>
void copyToHost(const T* device_input, size_t num_items, T* host_output) {
    cuMemcpyToHost(device_input, num_items, host_output);
}

template<typename T>
void copyToHostAsync(const T* device_input, size_t num_items, T* host_output) {
    cuMemcpyToHostAsync(device_input, num_items, host_output);
}

template<typename T>
void copyFromHost(const T* host_input, size_t num_items, T* device_output) {
    cuMemcpyToDevice(host_input, num_items, device_output);
}
/*
template<typename T>
void copyHostToDevice(T value, T* destination) {
    cuMemcpyToDevice(value, destination);
}

template<typename T>
void copyDeviceToHost(const T* source, size_t num_items, T* destination) {
    cuMemcpyToHost(source, num_items, destination);
}

template<typename T>
void copyDeviceToHost(const T* source, T& value) {
    cuMemcpyToHost(source, value);
}*/

template<typename T>
void memset(T* pointer, size_t num_items, unsigned char mask) {
    cuMemset(pointer, num_items, mask);
}

template<typename T>
void memsetZero(T* pointer, size_t num_items) {
    cuMemset0x00(pointer, num_items);
}

template<typename T>
void memsetOne(T* pointer, size_t num_items) {
    cuMemset0xFF(pointer, num_items);
}

template<typename T>
void excl_prefixsum(const T* input, size_t num_items, T* output) {
    xlib::CubExclusiveSum<T> cub_prefixsum(input, num_items, output);
    cub_prefixsum.run();
}

template<typename HostIterator, typename DeviceIterator>
bool equal(HostIterator host_start, HostIterator host_end,
           DeviceIterator device_start) noexcept {
    return xlib::gpu::equal(host_start, host_end, device_start);
}

template<typename T>
void printArray(const T* device_input, size_t num_items) {
    xlib::gpu::printArray(device_input, num_items);
}

} // namespace gpu

//==============================================================================

namespace host {

template<typename T>
void allocate(T*& pointer, size_t num_items) {
    pointer = new T[num_items];
}

template<typename T>
void allocatePageLocked(T*& pointer, size_t num_items) {
    cuMallocHost(pointer, num_items);
}

template<typename T>
void free(T*& pointer) {
    delete[] pointer;
}

template<typename T>
void freePageLocked(T*& pointer) {
    cuFreeHost(pointer);
}

template<typename... TArgs>
void freePageLocked(TArgs*... pointers) {
    cuFreeHost(pointers...);
}

template<typename T>
void copyToHost(const T* host_input, size_t num_items, T* host_output) {
    std::copy(host_input, host_input + num_items, host_output);
}

template<typename T>
void copyToDevice(const T* host_input, size_t num_items, T* device_output) {
    cuMemcpyToDevice(host_input, num_items, device_output);
}

template<typename T>
void copyToDeviceAsync(const T* host_input, size_t num_items, T* device_output) {
    cuMemcpyToDeviceAsync(host_input, num_items, device_output);
}

template<typename T>
void copyToDevice(T host_value, T* device_output) {
    cuMemcpyToDevice(host_value, device_output);
}

template<typename T>
void copyFromDevice(const T* device_input, size_t num_items, T* host_output) {
    cuMemcpyToHost(device_input, num_items, host_output);
}

template<typename T>
void copyFromDevice(const T* device_input, T& host_output) {
    cuMemcpyToHost(device_input, host_output);
}

template<typename T>
void memsetZero(T* pointer, size_t num_items) {
    std::memset(pointer, 0x0, num_items * sizeof(T));
}

template<typename T>
void memsetOne(T* pointer, size_t num_items) {
    std::memset(pointer, 0xFF, num_items * sizeof(T));
}

template<typename T>
void generate_randoms(T* pointer, size_t num_items, T min, T max) {
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch()
                .count();
    generate_randoms(pointer, num_items, min, max, seed);
}

template<typename T, typename S>
void generate_randoms(T* pointer, size_t num_items, T min, T max, S seed) {
    std::mt19937 engine(seed);
    std::uniform_int_distribution<T> distrib(min, max);
    std::generate(pointer, pointer + num_items,
                  [&](){ return distrib(engine); } );
}

template<typename T>
void printArray(const T* host_input, size_t num_items) {
    xlib::printArray(host_input, num_items);
}

} // namespace host
} // namespace hornet
