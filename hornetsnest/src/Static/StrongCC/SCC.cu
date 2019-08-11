/*
   * Copyright (c) 2019, NVIDIA CORPORATION.
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *     http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   */

// Author: Prasun Gera pgera@nvidia.com
#include <set>
#include "SCCOperators.cuh"

namespace hornets_nest {

StrongCC::StrongCC(HornetGraph& G_Reg, HornetGraph& G_Tran)
    : G_Reg(G_Reg), G_Tran(G_Tran), StaticAlgorithm(G_Reg), num_components(0) {
  gpu::allocate(d_component_labels, G_Reg.nV());
  cuMemset0xFF(d_component_labels, G_Reg.nV());
  gpu::allocate(d_num_components, 1);
  cuMemset0x00(d_num_components, 1);

  // TODO: Should create a copy of the input graphs since we are modifying
  // them. Hornet doesn't have a copy constructor
  assert(G_Reg.nV() == G_Tran.nV() && G_Reg.nE() == G_Tran.nE());
}

StrongCC::~StrongCC() {
  gpu::free(d_component_labels);
  gpu::free(d_num_components);
}

void StrongCC::reset() {}

void StrongCC::release() {}

bool StrongCC::validate() {
  return true;
}

// TODO: Currently the list is copied to the host. Leave it on device
// when we have a viable way of doing set intersections on device
// TODO: Replace with top-down + bottom up BFS since we have the transpose in
// SCC
void get_visited_vertices(HornetGraph& hornet,
                          vid_t root,
                          std::vector<vid_t>& h_visited) {
  int* d_visited;
  // TODO: Replace with a more efficient queue. We don't need two
  // levels. BFS just needs a single queue
  TwoLevelQueue<vid_t> queue;
  load_balancing::BinarySearch load_balancing(hornet);
  // Bitmap size on the device
  size_t d_visited_size = (hornet.nV() + INT_SIZE - 1) / INT_SIZE;
  gpu::allocate(d_visited, d_visited_size);
  // cudaMemset((void*)d_visited, 0, d_visited_size*sizeof(int));
  cuMemset0x00(d_visited, d_visited_size);
  queue.initialize(static_cast<size_t>(hornet.nV()));
  forAll(1, InitRootData{d_visited, root});

  // Host side call. Goes to the input-queue
  queue.insert(root);

  h_visited.resize(hornet.nV());

  size_t h_visited_off = 0;

  while (queue.size() > 0) {
    cuMemcpyToHost(
        queue.device_input_ptr(), queue.size(), &h_visited[h_visited_off]);
    h_visited_off += queue.size();
    forAllEdges(hornet, queue, BFSOperator{d_visited, queue}, load_balancing);
    queue.swap();
  };

  gpu::free(d_visited);
  h_visited.resize(h_visited_off);
}

eoff_t get_total_degree(HornetGraph& hornet,
                        const vid_t* d_vlist,
                        size_t vlist_size) {
  degree_t* d_total_deg;
  degree_t h_total_deg;
  gpu::allocate(d_total_deg, 1);
  cuMemset0x00(d_total_deg, 1);
  // FIXME: size_t is getting converted to int
  forAllVertices(hornet, d_vlist, vlist_size, GetTotalDeg{d_total_deg});
  cuMemcpyToHost(d_total_deg, 1, &h_total_deg);
  gpu::free(d_total_deg);
  return h_total_deg;
}

// src and dst pointers should have sufficient addressable memory (>=
// batch_size)
void create_batch_from_vlist(HornetGraph& hornet,
                             const vid_t* d_vlist,
                             const size_t vlist_size,
                             vid_t* d_batch_src,
                             vid_t* d_batch_dst,
                             const eoff_t batch_size) {
  if (!batch_size)
    return;
  // TODO: Replace with an efficient queue. Don't
  // need two levels
  TwoLevelQueue<vid2_t> src_dest_queue;
  src_dest_queue.initialize(static_cast<size_t>(batch_size));
  load_balancing::BinarySearch load_balancing(hornet);
  forAllEdges(hornet,
              d_vlist,
              vlist_size,
              MakeEdgePairs{src_dest_queue},
              load_balancing);

  // Since the queue was used on the GPU directly, use the output ptr
  forAll(batch_size,
         GetSrcIds{d_batch_src, src_dest_queue.device_output_ptr()});
  forAll(batch_size,
         GetDstIds{d_batch_dst, src_dest_queue.device_output_ptr()});
}

// Delete outgoing edges from the nodes in both the regular and the transposes
// graph
void delete_outgoing_edges(HornetGraph& G_Reg,
                           HornetGraph& G_Tran,
                           const vid_t* d_vlist,
                           const size_t vlist_size) {
  assert(G_Reg.nV() == G_Tran.nV() && G_Reg.nE() == G_Tran.nE());
  if (!G_Reg.nE() || !vlist_size)
    return;
  eoff_t batch_size = get_total_degree(G_Reg, d_vlist, vlist_size);
  if (!batch_size)
    return;
  vid_t *d_src = nullptr, *d_dst = nullptr;
  gpu::allocate(d_src, batch_size);
  gpu::allocate(d_dst, batch_size);
  create_batch_from_vlist(
      G_Reg, d_vlist, vlist_size, d_src, d_dst, batch_size);
  UpdatePtr ptr(batch_size, d_src, d_dst);
  Update batch_update(ptr);
  G_Reg.erase(batch_update);
  UpdatePtr tran_ptr(batch_size, d_dst, d_src);
  Update tran_batch_update(tran_ptr);
  G_Tran.erase(tran_batch_update);
  gpu::free(d_src);
  gpu::free(d_dst);
}

// Labels any remaining 0-degree nodes. Returns the number of newly labelled
// nodes
vid_t trim_isolated_nodes(HornetGraph& G_Reg,
                          HornetGraph& G_Tran,
                          vid_t* d_component_labels,
                          vid_t* d_num_components) {
  vid_t curr_comp_count, next_comp_count;
  cuMemcpyToHost(d_num_components, 1, &curr_comp_count);
  TwoLevelQueue<vid_t> zero_degree_queue;
  zero_degree_queue.initialize(static_cast<size_t>(G_Reg.nV()));
  // Label 0-outdegree nodes
  forAllVertices(G_Reg,
                 LabelZeroDegreeNodes{
                     d_num_components, d_component_labels, zero_degree_queue});
  // Delete incoming edges for 0-outdegree nodes (swap G_Reg and G_Tran args)
  delete_outgoing_edges(G_Tran,
                        G_Reg,
                        zero_degree_queue.device_output_ptr(),
                        zero_degree_queue.size_sync_out());
  // Reuse queue
  zero_degree_queue.swap();
  // Label 0-indegree nodes
  forAllVertices(G_Tran,
                 LabelZeroDegreeNodes{
                     d_num_components, d_component_labels, zero_degree_queue});
  // Delete outgoing edges for 0-indegree nodes
  delete_outgoing_edges(G_Reg,
                        G_Tran,
                        zero_degree_queue.device_output_ptr(),
                        zero_degree_queue.size_sync_out());
  cuMemcpyToHost(d_num_components, 1, &next_comp_count);
  return (next_comp_count - curr_comp_count);
}

void run_SCC(HornetGraph& G_Reg,
             HornetGraph& G_Tran,
             vid_t* d_component_labels,
             vid_t* d_num_components,
             vid_t& num_labelled_nodes) {
  if (num_labelled_nodes == G_Reg.nV())
    return;

  // TRIM phase: Label and delete 0 degree vertices
  // TODO: Should be optimised. It scans all nodes every time which is wasteful
  num_labelled_nodes +=
      trim_isolated_nodes(G_Reg, G_Tran, d_component_labels, d_num_components);

  /*
  std::cout << "G_Reg after trim:\n";
  G_Reg.print();
  std::cout << "G_Tran after trim:\n";
  G_Tran.print();
  */

  // If no edges left, return
  if (!G_Reg.nE())
    return;

  /*-----------------------------------------------------------*/
  /*               SCC = G_Desc ∩ G_Pred                       */
  /*-----------------------------------------------------------*/

  // Pick random starting node
  vid_t src = G_Reg.max_degree_id();
  std::vector<vid_t> visited_list, trans_visited_list;
  // TODO: Shouldn't need to get results back on host. Need GPU set
  // intersection.
  get_visited_vertices(G_Reg, src, visited_list);
  get_visited_vertices(G_Tran, src, trans_visited_list);
  std::sort(visited_list.begin(), visited_list.end());
  std::sort(trans_visited_list.begin(), trans_visited_list.end());

  std::vector<vid_t> scc_vlist;
  // TODO: Replace set intersection with hash graph when available
  std::set_intersection(visited_list.begin(),
                        visited_list.end(),
                        trans_visited_list.begin(),
                        trans_visited_list.end(),
                        std::back_inserter(scc_vlist));

  /*
  std::cout << "SCC:\n";
  for (auto v:scc_vlist){
          std::cout << v << " ";
  }
  std::cout << "\n";
  */

  vid_t *d_visited_list = nullptr, *d_tran_visited_list = nullptr,
        *d_scc_vlist = nullptr;
  if (visited_list.size()) {
    gpu::allocate(d_visited_list, visited_list.size());
    cuMemcpyToDevice(&visited_list.at(0), visited_list.size(), d_visited_list);
  }

  if (trans_visited_list.size()) {
    gpu::allocate(d_tran_visited_list, trans_visited_list.size());
    cuMemcpyToDevice(&trans_visited_list.at(0),
                     trans_visited_list.size(),
                     d_tran_visited_list);
  }

  if (scc_vlist.size()) {
    gpu::allocate(d_scc_vlist, scc_vlist.size());
    cuMemcpyToDevice(&scc_vlist.at(0), scc_vlist.size(), d_scc_vlist);
  }

  // Label SCC nodes
  if (scc_vlist.size()) {
    vid_t curr_comp_count;
    cuMemcpyToHost(d_num_components, 1, &curr_comp_count);
    forAllVertices(G_Reg,
                   d_scc_vlist,
                   scc_vlist.size(),
                   LabelNodes{curr_comp_count, d_component_labels});
    curr_comp_count++;
    cuMemcpyToDevice(&curr_comp_count, 1, d_num_components);
    num_labelled_nodes += scc_vlist.size();
  }

  /*-----------------------------------------------------------*/
  /*              G_Desc_new = G_Desc_orig \ SCC               */
  /*-----------------------------------------------------------*/

  // TODO: This is very verbose. It's not done inside a function because we
  // need to reuse d_g_desc_src and d_g_desc_dst later for creating G_Rem

  // G_Desc_Reg = Descendents from the BFS root
  vid_t *d_g_desc_src = nullptr, *d_g_desc_dst = nullptr;
  eoff_t g_desc_batch_size =
      get_total_degree(G_Reg, d_visited_list, visited_list.size());

  HornetGraph G_Desc_Reg(G_Reg.nV());
  HornetGraph G_Desc_Tran(G_Reg.nV());

  if (g_desc_batch_size) {
    gpu::allocate(d_g_desc_src, g_desc_batch_size);
    gpu::allocate(d_g_desc_dst, g_desc_batch_size);
    create_batch_from_vlist(G_Reg,
                            d_visited_list,
                            visited_list.size(),
                            d_g_desc_src,
                            d_g_desc_dst,
                            g_desc_batch_size);
    UpdatePtr g_desc_ptr(g_desc_batch_size, d_g_desc_src, d_g_desc_dst);
    Update g_desc_batch_update(g_desc_ptr);
    G_Desc_Reg.insert(g_desc_batch_update);

    // std::cout << "G_Desc_orig:\n";
    // G_Desc_Reg.print();

    // Now create a tranpose of G_Desc_Reg.
    // Same steps as above but with swapped ends
    UpdatePtr g_desc_tran_ptr(g_desc_batch_size, d_g_desc_dst, d_g_desc_src);
    Update g_desc_tran_batch_update(g_desc_tran_ptr);
    G_Desc_Tran.insert(g_desc_tran_batch_update);

    // std::cout << "Print G_Desc_Tran_orig\n";
    // G_Desc_Tran.print();

    // Compute G_Desc_Reg - SCC_Desc (and the transpose version)
    delete_outgoing_edges(
        G_Desc_Reg, G_Desc_Tran, d_scc_vlist, scc_vlist.size());

    // G_Desc_Reg is now G_Desc_Reg - SCC_Desc
    // std::cout << "G_Desc_new (G_Desc_orig - SCC):\n";
    // G_Desc_Reg.print();
    // G_Desc_Tran is now G_Desc_Tran - SCC_Desc_Tran
    // std::cout << "G_Desc_Tran_new (G_Desc_Tran_orig - SCC_Desc_Tran):\n";
    // G_Desc_Tran.print();

    // G_Desc_Reg and G_Desc_Tran will be passed to the recursive call
  }

  /*-----------------------------------------------------------*/
  /*            G_Pred_new = G_Pred_orig \ SCC                 */
  /*-----------------------------------------------------------*/

  // TODO: This is very verbose. It's not done inside a function because we
  // need to reuse d_g_pred_src and d_g_pred_dst later for creating G_Rem

  // G_Pred_Tran = Transpose of Predecessor edges (i.e., Descendents from the
  // BFS root in the transpose graph)
  vid_t *d_g_pred_tran_src = nullptr, *d_g_pred_tran_dst = nullptr;
  eoff_t g_pred_tran_batch_size =
      get_total_degree(G_Tran, d_tran_visited_list, trans_visited_list.size());

  HornetGraph G_Pred_Tran(G_Tran.nV());
  HornetGraph G_Pred_Reg(G_Tran.nV());
  if (g_pred_tran_batch_size) {
    gpu::allocate(d_g_pred_tran_src, g_pred_tran_batch_size);
    gpu::allocate(d_g_pred_tran_dst, g_pred_tran_batch_size);
    create_batch_from_vlist(G_Tran,
                            d_tran_visited_list,
                            trans_visited_list.size(),
                            d_g_pred_tran_src,
                            d_g_pred_tran_dst,
                            g_pred_tran_batch_size);
    UpdatePtr g_pred_tran_ptr(
        g_pred_tran_batch_size, d_g_pred_tran_src, d_g_pred_tran_dst);
    Update g_pred_tran_batch_update(g_pred_tran_ptr);
    G_Pred_Tran.insert(g_pred_tran_batch_update);
    // std::cout << "G_Pred_Tran_orig:\n";
    // G_Pred_Tran.print();

    // Now create a tranpose of G_Pred_Tran
    // Same steps as above but with swapped ends
    UpdatePtr g_pred_ptr(
        g_pred_tran_batch_size, d_g_pred_tran_dst, d_g_pred_tran_src);
    Update g_pred_batch_update(g_pred_ptr);
    G_Pred_Reg.insert(g_pred_batch_update);
    // std::cout << "G_Pred_Reg_orig:\n";
    // G_Pred_Reg.print();

    // Compute G_Pred_Tran - SCC_Tran (and the transpose version)
    delete_outgoing_edges(
        G_Pred_Tran, G_Pred_Reg, d_scc_vlist, scc_vlist.size());
    // G_Pred_Tran is now G_Pred_Tran - SCC_Pred_Tran
    // std::cout << "G_Pred_Tran_new (G_Pred_Tran_orig - SCC_Pred_Tran):\n";
    // G_Pred_Tran.print();

    // G_Pred_Reg is now G_Pred_Reg - SCC_Pred_Reg
    // std::cout << "G_Pred_Reg_new (G_Pred_Reg_orig - SCC_Pred_Reg):\n";
    // G_Pred_Reg.print();

    // G_Pred_Reg and G_Pred_Tran will be passed to the recursive call
  }

  /*-----------------------------------------------------------*/
  /*      G_Rem = G_orig \ (G_Pred_orig ∪ G_Desc_orig)         */
  /*-----------------------------------------------------------*/

  // G_Reg remainder = G_Reg - (descendents + predecessors)
  if (G_Reg.nE() && g_desc_batch_size) {
    UpdatePtr g_desc_ptr(g_desc_batch_size, d_g_desc_src, d_g_desc_dst);
    Update g_desc_batch_update(g_desc_ptr);
    // Delete G_Desc_Reg
    G_Reg.erase(g_desc_batch_update);
  }

  if (G_Reg.nE() && g_pred_tran_batch_size) {
    // Note: Swapped ends to get Pred_Reg
    UpdatePtr g_pred_ptr(
        g_pred_tran_batch_size, d_g_pred_tran_dst, d_g_pred_tran_src);
    Update g_pred_batch_update(g_pred_ptr);
    // Delete G_Pred_Reg
    G_Reg.erase(g_pred_batch_update);
  }

  // std::cout << "G_Rem_Reg (G_Reg - G_Pred_Reg_orig - G_Desc_Reg_orig):\n";
  // G_Reg.print();

  // G_Tran remainder = G_Tran - (descendents + predecessors)
  if (G_Tran.nE() && g_pred_tran_batch_size) {
    UpdatePtr g_pred_tran_ptr(
        g_pred_tran_batch_size, d_g_pred_tran_src, d_g_pred_tran_dst);
    Update g_pred_tran_batch_update(g_pred_tran_ptr);
    // Delete G_Pred_Tran
    G_Tran.erase(g_pred_tran_batch_update);
  }

  if (G_Tran.nE() && g_desc_batch_size) {
    // Note: Swapped ends to get Desc_Tran
    UpdatePtr g_desc_tran_ptr(g_desc_batch_size, d_g_desc_dst, d_g_desc_src);
    Update g_desc_tran_batch_update(g_desc_tran_ptr);
    // Delete G_Desc_Tran
    G_Tran.erase(g_desc_tran_batch_update);
  }

  // std::cout << "G_Rem_Tran (G_Tran - G_Pred_Tran_orig -
  // G_Desc_Tran_orig):\n";
  // G_Tran.print();
  // G_Reg and G_Tran will be passed to the recursive call

  // Reclaim working memory before recursion
  gpu::free(d_visited_list);
  gpu::free(d_tran_visited_list);
  gpu::free(d_scc_vlist);
  gpu::free(d_g_pred_tran_src);
  gpu::free(d_g_pred_tran_dst);
  gpu::free(d_g_desc_src);
  gpu::free(d_g_desc_dst);

  /*-----------------------------------------------------------*/
  /*                      Recursive calls                      */
  /*-----------------------------------------------------------*/

  run_SCC(G_Desc_Reg,
          G_Desc_Tran,
          d_component_labels,
          d_num_components,
          num_labelled_nodes);
  run_SCC(G_Pred_Reg,
          G_Pred_Tran,
          d_component_labels,
          d_num_components,
          num_labelled_nodes);
  run_SCC(
      G_Reg, G_Tran, d_component_labels, d_num_components, num_labelled_nodes);
}

void StrongCC::run() {
  vid_t num_labelled_nodes = 0;
  run_SCC(
      G_Reg, G_Tran, d_component_labels, d_num_components, num_labelled_nodes);
  assert(num_labelled_nodes == G_Reg.nV());
  vid_t curr_comp_count;
  cuMemcpyToHost(d_num_components, 1, &curr_comp_count);
  std::cout << "Num Components: " << curr_comp_count << "\n";
  std::vector<vid_t> component_labels(G_Reg.nV());
  cuMemcpyToHost(d_component_labels, G_Reg.nV(), &component_labels[0]);

  std::cout << "Vid\tComponent id\n";
  for (vid_t i = 0; i < G_Reg.nV(); i++) {
    std::cout << i << "\t" << component_labels[i] << "\n";
  }
}
}
