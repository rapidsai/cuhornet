#include "../SoA/SoAData.cuh"

#include <rmm/thrust_rmm_allocator.h>

using namespace rmm;

namespace hornet {
namespace gpu {

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
int HORNETSTATIC::_instance_count = 0;

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HORNETSTATIC::
HornetStatic(HORNETSTATIC::HInitT& h_init, DeviceType h_init_type) noexcept :
    _nV(h_init.nV()),
    _nE(h_init.nE()),
    _id(_instance_count++),
    _vertex_data(h_init.nV()),
    _edge_data(xlib::upper_approx<512>(h_init.nE())) {
    initialize(h_init, h_init_type);
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
HORNETSTATIC::
initialize(HORNETSTATIC::HInitT& h_init, DeviceType h_init_type) noexcept {
  if (h_init_type == DeviceType::HOST) {
    SoAData<VertexTypes, DeviceType::HOST> vertex_data(h_init.nV());
    auto e_d = vertex_data.get_soa_ptr();

    xlib::byte_t * edge_block_ptr =
        reinterpret_cast<xlib::byte_t *>(_edge_data.get_soa_ptr().template get<0>());

    const auto * offsets = h_init.csr_offsets();
    for (int i = 0; i < h_init.nV(); ++i) {
        auto degree = offsets[i + 1] - offsets[i];
        auto e_ref = e_d[i];
        e_ref.template get<0>() = degree;
        e_ref.template get<1>() = edge_block_ptr;
        e_ref.template get<2>() = offsets[i];
        e_ref.template get<3>() = _edge_data.get_num_items();
    }
    _vertex_data.template copy(vertex_data);
    _edge_data.copy(h_init.edge_data_ptr(), DeviceType::HOST, h_init.nE());
  } else {

    xlib::byte_t * edge_block_ptr =
        reinterpret_cast<xlib::byte_t *>(_edge_data.get_soa_ptr().template get<0>());
    auto num_items = _edge_data.get_num_items();

    auto degreePtr  = thrust::device_pointer_cast(_vertex_data.get_soa_ptr().template get<0>());
    auto blockPtr   = thrust::device_pointer_cast(_vertex_data.get_soa_ptr().template get<1>());
    auto offsetPtr  = thrust::device_pointer_cast(_vertex_data.get_soa_ptr().template get<2>());
    auto numItemPtr = thrust::device_pointer_cast(_vertex_data.get_soa_ptr().template get<3>());

    auto initOffset = thrust::device_pointer_cast(h_init.csr_offsets());
    thrust::transform(initOffset + 1, initOffset + 1 + h_init.nV(), initOffset, degreePtr, thrust::minus<degree_t>());
    thrust::fill(blockPtr, blockPtr + h_init.nV(), edge_block_ptr);
    thrust::copy(initOffset, initOffset + h_init.nV(), offsetPtr);
    thrust::fill(numItemPtr, numItemPtr + h_init.nV(), num_items);

    _edge_data.copy(h_init.edge_data_ptr(), DeviceType::DEVICE, h_init.nE());
  }
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
degree_t
HORNETSTATIC::
nV(void) const noexcept {
    return _nV;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
degree_t
HORNETSTATIC::
nE(void) const noexcept {
    return _nE;
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
HORNETSTATIC::HornetDeviceT
HORNETSTATIC::
device(void) noexcept {
    return HornetDeviceT(_nV, _nE, _vertex_data.get_soa_ptr());
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes,
    typename vid_t, typename degree_t>
void
HORNETSTATIC::
print(void) {
    SoAData<
        TypeList<degree_t, xlib::byte_t*, degree_t, degree_t, VertexMetaTypes...>,
        DeviceType::HOST> host_vertex_data(_vertex_data.get_num_items());
    host_vertex_data.copy(_vertex_data);
    auto ptr = host_vertex_data.get_soa_ptr();
    for (int i = 0; i < _nV; ++i) {
        degree_t v_degree = ptr[i].template get<0>();
        std::cout<<i<<" : "<<v_degree<<" | ";
        rmm::device_vector<degree_t> dst(v_degree);
        vid_t * dst_ptr = reinterpret_cast<vid_t*>(ptr[i].template get<1>()) + ptr[i].template get<2>();
        thrust::copy(dst_ptr, dst_ptr + v_degree, dst.begin());
        thrust::copy(dst.begin(), dst.end(), std::ostream_iterator<vid_t>(std::cout, " "));
        std::cout<<"\n";

    }
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes, typename vid_t, typename degree_t>
vid_t
HORNETSTATIC::
max_degree_id() const noexcept {
    auto start_ptr = _vertex_data.get_soa_ptr().template get<0>();
    cudaStream_t stream{nullptr};    
    auto* iter = thrust::max_element(rmm::exec_policy(stream)->on(stream), start_ptr, start_ptr + _nV);
    if (iter == start_ptr + _nV) {
        return static_cast<vid_t>(-1);
    } else {
        return static_cast<vid_t>(iter - start_ptr);
    }
}

template <typename... VertexMetaTypes, typename... EdgeMetaTypes, typename vid_t, typename degree_t>
degree_t
HORNETSTATIC::
max_degree() const noexcept {
    auto start_ptr = _vertex_data.get_soa_ptr().template get<0>();
    cudaStream_t stream{nullptr};

    auto* iter = thrust::max_element(rmm::exec_policy(stream)->on(stream), start_ptr, start_ptr + _nV);
    if (iter == start_ptr + _nV) {
        return static_cast<degree_t>(0);
    } else {
        degree_t d;
        cudaMemcpy(&d, iter, sizeof(degree_t), cudaMemcpyDeviceToHost);
        return d;
    }
}

}
}
