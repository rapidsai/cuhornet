/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date September, 2017
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
 */

#include <rmm/thrust_rmm_allocator.h>

using namespace rmm;

namespace hornet {

//==============================================================================
/////////////////////////
// AllocateTupleVector //
/////////////////////////

template<int N, int SIZE>
struct AllocateTupleVector {
    template <typename... Ts>
    static void allocate(std::tuple<Ts...>& c, const size_t num_items) {
        std::get<N>(c).clear();
        std::get<N>(c).resize(num_items);
        AllocateTupleVector<N+1, SIZE>::allocate(c, num_items);
    }
};

template<int N>
struct AllocateTupleVector<N, N> {
    template <typename... Ts>
    static void allocate(std::tuple<Ts...>& c, const size_t num_items) {
    }
};

template <typename... Ts>
void allocate_tuple_vector(std::tuple<Ts...>& c, const size_t num_items) {
  AllocateTupleVector<0, sizeof...(Ts)>::
    allocate(c, num_items);
}

//==============================================================================
///////////////
// SetSoAPtr //
///////////////

template <typename T>
T*
get_ptr(rmm::device_vector<T>& t) {
  return t.data().get();
}

template <typename T>
T*
get_ptr(thrust::host_vector<T>& t) {
  return t.data();
}

template<int N, int SIZE>
struct SetSoAPtr {
    template <typename T1, typename T2>
    static void set(T1& c, T2& ptr) {
        ptr.template set<N>(get_ptr(std::get<N>(c)));
        SetSoAPtr<N+1, SIZE>::set(c, ptr);
    }
};

template<int N>
struct SetSoAPtr<N, N> {
    template <typename T1, typename T2>
    static void set(T1& c, T2& ptr) {
    }
};

template <typename T1, typename T2>
void set_soa_ptr(T1& c, T2& ptr) {
    SetSoAPtr<0, std::tuple_size<T1>::value>::set(c, ptr);
}

template <typename... Ts>
SoAPtr<Ts...> allocate_data(std::tuple<Ts...>& c, const size_t num_items) {
  AllocateTupleVector<0, sizeof...(Ts)>::
    allocate(c, num_items);
  SoAPtr<Ts...> ptr;
  set_soa_ptr(c, ptr);
  return ptr;
}

//==============================================================================
/////////////////
// TupleResize //
/////////////////

template<int N, int SIZE>
struct TupleResize {
    template <typename... Ts>
    static void resize(std::tuple<Ts...>& c, const size_t num_items) {
        std::get<N>(c).resize(num_items);
        TupleResize<N+1, SIZE>::resize(c, num_items);
    }
};

template<int N>
struct TupleResize<N, N> {
    template <typename... Ts>
    static void resize(std::tuple<Ts...>& c, const size_t num_items) {
    }
};

template <typename... Ts>
void tuple_resize(std::tuple<Ts...>& c, const size_t num_items) {
  TupleResize<0, sizeof...(Ts)>::
    resize(c, num_items);
}

//==============================================================================
//////////////////////
// RecursiveSetNull //
//////////////////////

template<int N, int SIZE>
struct RecursiveSetNull {
    template<template <typename...> typename Contnr, typename... Ts>
    static void set_null(Contnr<Ts...>& c) {
        c.template set<N>(nullptr);
        RecursiveSetNull<N+1, SIZE>::set_null(c);
    }
};

template<int N>
struct RecursiveSetNull<N, N> {
    template<template <typename...> typename Contnr, typename... Ts>
    static void set_null(Contnr<Ts...>& c) {
        c.template set<N>(nullptr);
    }
};

//==============================================================================
///////////////////
// RecursivePrint //
///////////////////

struct DevicePrint {
    template<typename T>
    static void print(
            T * src,
            DeviceType src_device_type,
            int num_items) {
        if (src_device_type == DeviceType::DEVICE) {
          //std::cout<<"TODO : Implement\n";
          auto sptr = thrust::device_pointer_cast(src);
          thrust::copy(sptr, sptr + num_items, std::ostream_iterator<T>(std::cout, " "));
        } else if (src_device_type == DeviceType::HOST) {
            for (int i = 0; i < num_items; ++i) {
                std::cout<<src[i]<<" ";
            }
        }
    }
};

template<int N, int SIZE>
struct RecursivePrint {
    template<
        template <typename...> typename SrcContnr,
        typename... Ts>
    static void print(
            const SrcContnr<Ts...>& src,
            const DeviceType src_device_type,
            const int num_items) {
        std::cout<<"N : "<<N<<" | ";
        DevicePrint::print(
                src.template get<N>(), src_device_type,
                num_items);
        std::cout<<"\n";
        RecursivePrint<N + 1, SIZE>::print(src, src_device_type, num_items);
    }

    template<
        template <typename...> typename SrcContnr,
        typename... Ts>
    static void print(
            const SrcContnr<Ts const...>& src,
            const DeviceType src_device_type,
            const int num_items) {
        std::cout<<"N : "<<N<<" | ";
        DevicePrint::print(
                src.template get<N>(), src_device_type,
                num_items);
        std::cout<<"\n";
        RecursivePrint<N + 1, SIZE>::print(src, src_device_type, num_items);
    }
};

template<int N>
struct RecursivePrint<N, N> {
    template<
        template <typename...> typename SrcContnr,
        typename... Ts>
    static void print(
            const SrcContnr<Ts...>& src,
            const DeviceType src_device_type,
            const int num_items) {
        std::cout<<"N : "<<N<<" | ";
        DevicePrint::print(
                src.template get<N>(), src_device_type,
                num_items);
        std::cout<<"\n";
    }

    template<
        template <typename...> typename SrcContnr,
        typename... Ts>
    static void print(
            const SrcContnr<Ts const...>& src,
            const DeviceType src_device_type,
            const int num_items) {
        std::cout<<"N : "<<N<<" | ";
        DevicePrint::print(
                src.template get<N>(), src_device_type,
                num_items);
        std::cout<<"\n";
    }
};

//==============================================================================
/////////////
// SoAData //
/////////////

template<typename... Ts, DeviceType device_t>
SoAData<TypeList<Ts...>, device_t>::
SoAData(const int num_items, bool initToZero) noexcept :
_num_items(num_items), _capacity(num_items) {
    if (num_items != 0) {
        allocate_tuple_vector(_data, static_cast<size_t>(_capacity));
        set_soa_ptr(_data, _soa);

        if (initToZero) {
          setEmpty();
        }
    }
}

template<typename... Ts, DeviceType device_t>
template<DeviceType d_t>
SoAData<TypeList<Ts...>, device_t>::
SoAData(SoAData<TypeList<Ts...>, d_t>&& other) noexcept :
_num_items(other._num_items), _capacity(other._capacity) {
    if (d_t != device_t) {
        //If other's device type is different, do not alter its variables

        allocate_tuple_vector(_data, static_cast<size_t>(_capacity));
        set_soa_ptr(_data, _soa);
        RecursiveCopy<0, sizeof...(Ts) - 1>::copy(other._soa, d_t, _soa, device_t, _num_items);
    } else {
        _data = std::move(other._data);
        RecursiveSetNull<0, sizeof...(Ts) - 1>::set_null(other._soa);
        other._num_items = 0;
        other._capacity = 0;
    }
    set_soa_ptr(_data, _soa);
}

template<typename... Ts, DeviceType device_t>
SoAData<TypeList<Ts...>, device_t>::
~SoAData(void) noexcept {
}

template<typename... Ts, DeviceType device_t>
SoAData<TypeList<Ts...>, device_t>&
SoAData<TypeList<Ts...>, device_t>::
operator=(SoAData<TypeList<Ts...>, device_t>&& other) {
    _data = std::move(other._data);
    RecursiveSetNull<0, sizeof...(Ts) - 1>::set_null(other._soa);
    _num_items = other._num_items;
    _capacity = other._capacity;
    other._num_items = 0;
    other._capacity = 0;
    return *this;
}

template<typename... Ts, DeviceType device_t>
SoAPtr<Ts...>&
SoAData<TypeList<Ts...>, device_t>::
get_soa_ptr(void) noexcept {
    return _soa;
}

template<typename... Ts, DeviceType device_t>
const SoAPtr<Ts...>&
SoAData<TypeList<Ts...>, device_t>::
get_soa_ptr(void) const noexcept {
    return _soa;
}

template<typename... Ts, DeviceType device_t>
template<DeviceType d_t>
void
SoAData<TypeList<Ts...>, device_t>:://TODO remove get_soa_ptr because SoAData<typename, DeviceType> is friend
copy(const SoAData<TypeList<Ts...>, d_t>& other) noexcept {
    int _item_count = std::min(other._num_items, _num_items);
    RecursiveCopy<0, sizeof...(Ts) - 1>::copy(other._soa, d_t, _soa, device_t, _item_count);
}

template<typename... Ts, DeviceType device_t>
void
SoAData<TypeList<Ts...>, device_t>:://TODO remove get_soa_ptr because SoAData<typename, DeviceType> is friend
copy(SoAPtr<Ts...> other, const DeviceType other_d_t, const int other_num_items) noexcept {
    int _item_count = std::min(other_num_items, _num_items);
    RecursiveCopy<0, sizeof...(Ts) - 1>::copy(other._soa, other_d_t, _soa, device_t, _item_count);
}

template<typename... Ts, DeviceType device_t>
template<DeviceType d_t>
void
SoAData<TypeList<Ts...>, device_t>:://TODO remove get_soa_ptr because SoAData<typename, DeviceType> is friend
append(const SoAData<TypeList<Ts...>, d_t>& other) noexcept {
  auto old_num_items = _num_items;
  resize(_num_items + other._num_items);
  RecursiveCopy<0, sizeof...(Ts) - 1>::copy(other._soa, d_t, _soa, device_t, other._num_items, 0, old_num_items);
}

template<typename... Ts, DeviceType device_t>
void
SoAData<TypeList<Ts...>, device_t>::
sort(void) noexcept {
  rmm::device_vector<int> range;
  if (sizeof...(Ts) > 3) {

    std::tuple<VectorType<Ts>...> temp_data;
    allocate_tuple_vector(temp_data, static_cast<size_t>(_num_items));
    SoAPtr<Ts...> temp_soa;
    set_soa_ptr(temp_data, temp_soa);

    sort_batch(_soa, _num_items, range, temp_soa);

    _data = std::move(temp_data);
    _soa = temp_soa;
  } else {
    SoAPtr<Ts...> temp_soa;
    sort_batch(_soa, _num_items, range, temp_soa);
  }
}

template<typename... Ts, DeviceType device_t>
template <typename degree_t>
void
SoAData<TypeList<Ts...>, device_t>::
gather(SoAData<TypeList<Ts...>, device_t>& other, const Map<degree_t>& map) noexcept {
  RecursiveGather<0, sizeof...(Ts)>::assign(other._soa, _soa, map, static_cast<degree_t>(map.size()));
}

template<typename... Ts, DeviceType device_t>
int
SoAData<TypeList<Ts...>, device_t>::
get_num_items(void) noexcept {
    return _num_items;
}

template<typename... Ts, DeviceType device_t>
void
SoAData<TypeList<Ts...>, device_t>::
resize(const int resize_items) noexcept {
    tuple_resize(_data, resize_items);
    set_soa_ptr(_data, _soa);
    if (resize_items > _capacity) {
        _capacity = resize_items;
    }
    _num_items = resize_items;
}

template<typename... Ts, DeviceType device_t>
DeviceType
SoAData<TypeList<Ts...>, device_t>::
get_device_type(void) noexcept {
    return device_t;
}

template<typename... Ts, DeviceType device_t>
void
SoAData<TypeList<Ts...>, device_t>::
setEmpty(void) noexcept {
  RecursiveFillNull<0, sizeof...(Ts)>::fillNull(_soa, _capacity, device_t);
}

//==============================================================================
//////////////
// CSoAData //
//////////////

template<typename... Ts, DeviceType device_t>
CSoAData<TypeList<Ts...>, device_t>::
CSoAData(const int num_items) noexcept :
_num_items(num_items), _capacity(xlib::upper_approx<512>(num_items)),
_data(xlib::SizeSum<Ts...>::value * _capacity),
_soa(get_ptr(_data), _capacity) {}

template<typename... Ts, DeviceType device_t>
CSoAData<TypeList<Ts...>, device_t>::
CSoAData(CSoAData<TypeList<Ts...>, device_t>&& other) noexcept :
_num_items(other._num_items), _capacity(xlib::upper_approx<512>(_num_items)),
_data(std::move(other._data)),
_soa(nullptr, 0) {
    _soa = other._soa;
    other._soa = CSoAPtr<Ts...>(nullptr, 0);
    other._num_items = 0;
    other._capacity = 0;
}

template<typename... Ts, DeviceType device_t>
CSoAData<TypeList<Ts...>, device_t>&
CSoAData<TypeList<Ts...>, device_t>::
operator=(CSoAData<TypeList<Ts...>, device_t>&& other) {
    _data = std::move(other._data);
    _soa = other._soa;
    other._soa = CSoAPtr<Ts...>(nullptr, 0);
    other._num_items = 0;
    other._capacity = 0;
    return *this;
}

template<typename... Ts, DeviceType device_t>
CSoAData<TypeList<Ts...>, device_t>::
~CSoAData(void) noexcept {
}

template<typename... Ts, DeviceType device_t>
CSoAPtr<Ts...>&
CSoAData<TypeList<Ts...>, device_t>::
get_soa_ptr(void) noexcept {
    return _soa;
}

template<typename... Ts, DeviceType device_t>
const CSoAPtr<Ts...>&
CSoAData<TypeList<Ts...>, device_t>::
get_soa_ptr(void) const noexcept {
    return _soa;
}

template<typename... Ts, DeviceType device_t>
void
CSoAData<TypeList<Ts...>, device_t>::
copy(SoAPtr<Ts const...> other, DeviceType other_d_t, int other_num_items) noexcept {
    int _item_count = std::min(other_num_items, _num_items);
    RecursiveCopy<0, sizeof...(Ts) - 1>::copy(other, other_d_t, _soa, device_t, _item_count);
}

template<typename... Ts, DeviceType device_t>
void
CSoAData<TypeList<Ts...>, device_t>::
copy(SoAPtr<Ts...> other, DeviceType other_d_t, int other_num_items) noexcept {
    int _item_count = std::min(other_num_items, _num_items);
    RecursiveCopy<0, sizeof...(Ts) - 1>::copy(other, other_d_t, _soa, device_t, _item_count);
}

template<typename... Ts, DeviceType device_t>
void
CSoAData<TypeList<Ts...>, device_t>::
copy(CSoAPtr<Ts...> other, DeviceType other_d_t, int other_num_items) noexcept {
    int _item_count = std::min(other_num_items, _num_items);
    RecursiveCopy<0, sizeof...(Ts) - 1>::copy(other, other_d_t, _soa, device_t, _item_count);
}

template<typename... Ts, DeviceType device_t>
template<DeviceType d_t>
void
CSoAData<TypeList<Ts...>, device_t>::
copy(CSoAData<TypeList<Ts...>, d_t>&& other) noexcept {
    //Copy the whole block as is
    if ((other._capacity == _capacity) &&
            (other._capacity == other._num_items) &&
            (other._num_items == _num_items)) {
        DeviceCopy::copy(
                reinterpret_cast<xlib::byte_t const *>(
                    other.get_soa_ptr().template get<0>()), d_t,
                reinterpret_cast<xlib::byte_t*>(
                    _soa.template get<0>()), device_t,
                xlib::SizeSum<Ts...>::value * other._capacity);
    } else {
        int _item_count = std::min(other._num_items, _num_items);
        RecursiveCopy<0, sizeof...(Ts) - 1>::copy(
                other.get_soa_ptr(), d_t, _soa, device_t, _item_count);
    }
}

template<typename... Ts, DeviceType device_t>
int
CSoAData<TypeList<Ts...>, device_t>::
get_num_items(void) noexcept {
    return _num_items;
}

template<typename... Ts, DeviceType device_t>
void
CSoAData<TypeList<Ts...>, device_t>::
resize(const int resize_items) noexcept {
    int new_capacity = xlib::upper_approx<512>(resize_items);
    if (new_capacity > _capacity) {

        BufferType temp_data(xlib::SizeSum<Ts...>::value * new_capacity);
        CSoAPtr<Ts...> temp_soa(get_ptr(temp_data), new_capacity);

        RecursiveCopy<0, sizeof...(Ts) - 1>::copy(
                _soa, device_t, temp_soa, device_t, _num_items);
        _data = std::move(temp_data);
        _soa = temp_soa;
        _capacity = new_capacity;
    }
    _num_items = resize_items;
}

template<typename... Ts, DeviceType device_t>
DeviceType
CSoAData<TypeList<Ts...>, device_t>::
get_device_type(void) noexcept {
    return device_t;
}

//==============================================================================
//print data


template<typename... Ts>
void print_soa(SoAData<TypeList<Ts...>, DeviceType::HOST>& data) {
    auto ptr = data.get_soa_ptr();
    RecursivePrint<0, sizeof...(Ts) - 1>::print(ptr, DeviceType::HOST, data.get_num_items());
}

template<typename... Ts>
void print_soa(SoAData<TypeList<Ts...>, DeviceType::DEVICE>& data) {
    auto ptr = data.get_soa_ptr();
    RecursivePrint<0, sizeof...(Ts) - 1>::print(ptr, DeviceType::DEVICE, data.get_num_items());
}

template<typename... Ts>
void print_soa(CSoAData<TypeList<Ts...>, DeviceType::HOST>& data) {
    auto ptr = data.get_soa_ptr();
    RecursivePrint<0, sizeof...(Ts) - 1>::print(ptr, DeviceType::HOST, data.get_num_items());
}

template<typename... Ts>
void print_soa(CSoAData<TypeList<Ts...>, DeviceType::DEVICE>& data) {
    auto ptr = data.get_soa_ptr();
    RecursivePrint<0, sizeof...(Ts) - 1>::print(ptr, DeviceType::DEVICE, data.get_num_items());
}

//==============================================================================

}
