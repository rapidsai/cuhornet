/**
 * @internal
 * @brief Vec-Tree interface
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
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

#include <limits>
#include <utility>

#include "Host/Basic.hpp"//xlib::byte_t

#include "Device/Util/SafeCudaAPI.cuh"
#include "Device/Util/SafeCudaAPISync.cuh"
#include "Device/Util/VectorUtil.cuh"
#include "Host/Numeric.hpp"

#include <cub/cub.cuh>
#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

namespace xlib {

class CubWrapper {
protected:
    void initialize(const int num_items) noexcept { _num_items = num_items; }
    void release(void) noexcept { _num_items = 0; }

    explicit CubWrapper() = default;
    explicit CubWrapper(const int num_items) noexcept : _num_items(num_items) {}
    ~CubWrapper() noexcept { release(); }

    mutable rmm::device_buffer _d_temp_storage { 0 };
    size_t _temp_storage_bytes { 0 };
    int    _num_items          { 0 };
};

//==============================================================================

//------------------------------------------------------------------------------

template<typename T, typename R>
class CubSortByKey : public CubWrapper {
public:
    explicit CubSortByKey() = default;

    explicit CubSortByKey(const int max_items) noexcept;

    void initialize(const int max_items) noexcept;

    void resize(const int max_items) noexcept;

    void shrink_to_fit(const int max_items) noexcept;

    void run(const T* d_key, const R* d_data_in, const int num_items,
             T* d_key_sorted, R* d_data_out,
             T d_key_max = std::numeric_limits<T>::max()) noexcept;

    static void srun(const T* d_key, const R* d_data_in, const int num_items,
                     T* d_key_sorted, R* d_data_out,
                     T d_key_max = std::numeric_limits<T>::max()) noexcept;
};

//==============================================================================

namespace cub_runlength {

template<typename T>
extern int run(const T* d_in, int num_items, T* d_unique_out,
               int* d_counts_out);

} // namespace cub_runlength

//------------------------------------------------------------------------------

template<typename T>
class CubRunLengthEncode : public CubWrapper {
public:
    explicit CubRunLengthEncode() = default;

    explicit CubRunLengthEncode(const int max_items) noexcept;

    ~CubRunLengthEncode() noexcept;

    void initialize(const int max_items) noexcept;

    void resize(const int max_items) noexcept;

    void release(void) noexcept;

    void shrink_to_fit(const int max_items) noexcept;

    int run(const T* d_in, const int num_items, T* d_unique_out, int* d_counts_out)
            noexcept;

    static int srun(const T* d_in, const int num_items, T* d_unique_out,
                    int* d_counts_out) noexcept;
private:
    rmm::device_vector<int> _d_num_runs_out;
};

//------------------------------------------------------------------------------

template<typename T>
class CubExclusiveSum : public CubWrapper {
public:
    explicit CubExclusiveSum(void) noexcept;

    explicit CubExclusiveSum(const int max_items) noexcept;

    void initialize(const int max_items) noexcept;

    void resize(const int max_items) noexcept;

    void shrink_to_fit(const int max_items) noexcept;

    void run(const T* d_in, const int num_items, T* d_out) const noexcept;

    void run(T* d_in_out, const int num_items) const noexcept;

    static void srun(const T* d_in, const int num_items, T* d_out) noexcept;

    static void srun(T* d_in_out, const int num_items) noexcept;
};

//==============================================================================

template<typename T>
class CubInclusiveMax : public CubWrapper {
public:
    explicit CubInclusiveMax() noexcept;

    explicit CubInclusiveMax(const int max_items) noexcept;

    void initialize(const int max_items) noexcept;

    void resize(const int max_items) noexcept;

    void shrink_to_fit(const int max_items) noexcept;

    void run(const T* d_in, const int num_items, T* d_out) const noexcept;

    void run(T* d_in_out, const int num_items) const noexcept;

    static void srun(const T* d_in, const int num_items, T* d_out) noexcept;

    static void srun(T* d_in_out, const int num_items) noexcept;
};

//==============================================================================

} // namespace xlib

namespace xlib {

///////////////
// SortByKey //
///////////////

template<typename T, typename R>
CubSortByKey<T, R>::CubSortByKey(const int max_items) noexcept  {
    initialize(max_items);
}

template<typename T, typename R>
void CubSortByKey<T, R>::initialize(const int max_items) noexcept {
    CubWrapper::initialize(max_items);
    size_t temp_storage_bytes = 0;
    T* d_key = nullptr, *d_key_sorted = nullptr;
    R* d_data_in = nullptr, *d_data_out = nullptr;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                    d_key, d_key_sorted,
                                    d_data_in, d_data_out,
                                    _num_items, 0, sizeof(T) * 8);
    _d_temp_storage.resize(temp_storage_bytes);
}

//------------------------------------------------------------------------------

template<typename T, typename R>
void CubSortByKey<T, R>::resize(const int max_items) noexcept {
    if (_num_items < max_items) {
        CubWrapper::release();
        initialize(max_items);
    }
}

template<typename T, typename R>
void CubSortByKey<T, R>::shrink_to_fit(const int max_items) noexcept {
    if (_num_items > max_items) {
        CubWrapper::release();
        initialize(max_items);
    }
}

//------------------------------------------------------------------------------

template<typename T, typename R>
void CubSortByKey<T, R>::run(
        const T* d_key,
        const R* d_data_in,
        const int num_items,
        T* d_key_sorted,
        R* d_data_out,
        T d_key_max) noexcept {
    int temp_num_items = num_items;
    using U = typename std::conditional<std::is_floating_point<T>::value,
                                        int, T>::type;
    int num_bits = std::is_floating_point<T>::value ? sizeof(T) * 8 :
                                     xlib::ceil_log2(static_cast<U>(d_key_max));
    cub::DeviceRadixSort::SortPairs(nullptr, _temp_storage_bytes,
                                    d_key, d_key_sorted,
                                    d_data_in, d_data_out,
                                    temp_num_items, 0, num_bits);
    cub::DeviceRadixSort::SortPairs(_d_temp_storage.data(), _temp_storage_bytes,
                                    d_key, d_key_sorted,
                                    d_data_in, d_data_out,
                                    temp_num_items, 0, num_bits);
}

//------------------------------------------------------------------------------

template<typename T, typename R>
void CubSortByKey<T, R>::srun(
        const T* d_key, const R* d_data_in,
        const int num_items, T* d_key_sorted,
        R* d_data_out, T d_key_max) noexcept {
    CubSortByKey<T, R> cub_instance(num_items);
    cub_instance.run(d_key, d_data_in, num_items, d_key_sorted, d_data_out);
}

//------------------------------------------------------------------------------

//==============================================================================
//==============================================================================
/////////////////////
// RunLengthEncode //
/////////////////////

namespace cub_runlength {

template<typename T>
int run(const T* d_in, const int num_items, T* d_unique_out,
        int* d_counts_out) {

    CubRunLengthEncode<T> cub_instance(num_items);
    return cub_instance.run(d_in, num_items, d_unique_out, d_counts_out);
}

template int run<int>(const int*, const int, int*, int*);

} // namespace cub_runlength

//------------------------------------------------------------------------------

template<typename T>
CubRunLengthEncode<T>::CubRunLengthEncode(const int max_items) noexcept {
    initialize(max_items);
}

template<typename T>
CubRunLengthEncode<T>::~CubRunLengthEncode() noexcept {
    release();
}

template<typename T>
void CubRunLengthEncode<T>::initialize(const int max_items) noexcept {
    CubWrapper::initialize(max_items);
    _d_num_runs_out.resize(1);
    T* d_in = nullptr, *d_unique_out = nullptr;
    int* d_counts_out = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, temp_storage_bytes,
                                       d_in, d_unique_out, d_counts_out,
                                       _d_num_runs_out.data().get(), _num_items);
    _d_temp_storage.resize(temp_storage_bytes);
}

//------------------------------------------------------------------------------

template<typename T>
void CubRunLengthEncode<T>::resize(const int max_items) noexcept {
    if (_num_items < max_items) {
        release();
        initialize(max_items);
    }
}

template<typename T>
void CubRunLengthEncode<T>::release(void) noexcept {
    CubWrapper::release();
}

template<typename T>
void CubRunLengthEncode<T>::shrink_to_fit(const int max_items) noexcept {
    if (_num_items > max_items) {
        release();
        initialize(max_items);
    }
}

//------------------------------------------------------------------------------

template<typename T>
int CubRunLengthEncode<T>::run(const T* d_in, const int num_items,
                               T* d_unique_out, int* d_counts_out) noexcept {
    int temp_num_items = num_items;
    size_t temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, temp_storage_bytes,
                                       d_in, d_unique_out, d_counts_out,
                                       _d_num_runs_out.data().get(), temp_num_items);
    cub::DeviceRunLengthEncode::Encode(_d_temp_storage.data(), temp_storage_bytes,
                                       d_in, d_unique_out, d_counts_out,
                                       _d_num_runs_out.data().get(), temp_num_items);
    return _d_num_runs_out[0];
}

//------------------------------------------------------------------------------

template<typename T>
int CubRunLengthEncode<T>::srun(const T* d_in, const int num_items, T* d_unique_out,
                                int* d_counts_out) noexcept {
    CubRunLengthEncode<T> cub_instance(num_items);
    return cub_instance.run(d_in, num_items, d_unique_out, d_counts_out);
}

template<typename T>
CubExclusiveSum<T>::CubExclusiveSum(void) noexcept {
}

template<typename T>
CubExclusiveSum<T>::CubExclusiveSum(const int max_items) noexcept {
    initialize(max_items);
}

template<typename T>
void CubExclusiveSum<T>::initialize(const int max_items) noexcept {
    CubWrapper::initialize(max_items);
    size_t temp_storage_bytes = 0;
    T* d_in = nullptr, *d_out = nullptr;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                  d_in, d_out, _num_items);
    if (temp_storage_bytes)
        _d_temp_storage.resize(temp_storage_bytes);
}

//------------------------------------------------------------------------------

template<typename T>
void CubExclusiveSum<T>::resize(const int max_items) noexcept {
    if (_num_items < max_items) {
        release();
        initialize(max_items);
    }
}

template<typename T>
void CubExclusiveSum<T>::shrink_to_fit(const int max_items) noexcept {
    if (_num_items > max_items) {
        release();
        initialize(max_items);
    }
}

//------------------------------------------------------------------------------

template<typename T>
void CubExclusiveSum<T>::run(
        const T* d_in,
        const int num_items,
        T* d_out) const noexcept {
    int temp_num_items = num_items;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                  d_in, d_out, temp_num_items);
    cub::DeviceScan::ExclusiveSum(_d_temp_storage.data(), temp_storage_bytes,
                                  d_in, d_out, temp_num_items);
}

template<typename T>
void CubExclusiveSum<T>::run(T* d_in_out, const int num_items) const noexcept {
    run(d_in_out, num_items, d_in_out);
}

//------------------------------------------------------------------------------

template<typename T>
void CubExclusiveSum<T>::srun(const T* d_in, const int num_items, T* d_out) noexcept {
    CubExclusiveSum<T> cub_instance(num_items);
    cub_instance.run(d_in, num_items, d_out);
}

template<typename T>
void CubExclusiveSum<T>::srun(T* d_in_out, const int num_items) noexcept {
    CubExclusiveSum::srun(d_in_out, num_items, d_in_out);
}

//==============================================================================
//==============================================================================
/////////////////////
// CubInclusiveMax //
/////////////////////

// CubMax functor
struct CubMax
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b > a) ? b : a;
    }
};

template<typename T>
CubInclusiveMax<T>::CubInclusiveMax(void) noexcept {
}

template<typename T>
CubInclusiveMax<T>::CubInclusiveMax(const int max_items) noexcept {
    initialize(max_items);
}

template<typename T>
void CubInclusiveMax<T>::initialize(const int max_items) noexcept {
    CubMax max_op;
    CubWrapper::initialize(max_items);
    size_t temp_storage_bytes = 0;
    T* d_in = nullptr, *d_out = nullptr;
    cub::DeviceScan::InclusiveScan(nullptr, temp_storage_bytes,
                                  d_in, d_out, max_op, _num_items);
    if (temp_storage_bytes)
        _d_temp_storage.resize(temp_storage_bytes);
}

//------------------------------------------------------------------------------

template<typename T>
void CubInclusiveMax<T>::resize(const int max_items) noexcept {
    if (_num_items < max_items) {
        release();
        initialize(max_items);
    }
}

template<typename T>
void CubInclusiveMax<T>::shrink_to_fit(const int max_items) noexcept {
    if (_num_items > max_items) {
        release();
        initialize(max_items);
    }
}

//------------------------------------------------------------------------------

template<typename T>
void CubInclusiveMax<T>::run(
        const T* d_in,
        const int num_items,
        T* d_out) const noexcept {
    CubMax max_op;
    int temp_num_items = num_items;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveScan(nullptr, temp_storage_bytes,
                                  d_in, d_out, max_op, temp_num_items);
    cub::DeviceScan::InclusiveScan(_d_temp_storage.data(), temp_storage_bytes,
                                  d_in, d_out, max_op, temp_num_items);
}

template<typename T>
void CubInclusiveMax<T>::run(T* d_in_out, const int num_items) const noexcept {
    run(d_in_out, num_items, d_in_out);
}

//------------------------------------------------------------------------------

template<typename T>
void CubInclusiveMax<T>::srun(const T* d_in, const int num_items, T* d_out) noexcept {
    CubInclusiveMax<T> cub_instance(num_items);
    cub_instance.run(d_in, num_items, d_out);
}

template<typename T>
void CubInclusiveMax<T>::srun(T* d_in_out, const int num_items) noexcept {
    CubInclusiveMax::srun(d_in_out, num_items, d_in_out);
}

} // namespace xlib
