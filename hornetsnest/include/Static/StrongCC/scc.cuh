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

#pragma once

#include "HornetAlg.hpp"

namespace hornets_nest {

using vid_t = int;
using HornetGraph = ::hornet::gpu::Hornet<vid_t>;
using HornetInit = ::hornet::HornetInit<vid_t>;

class StrongCC : public StaticAlgorithm<HornetGraph> {
 public:
  StrongCC(HornetGraph& G_Reg, HornetGraph& G_Tran);

  ~StrongCC();

  void reset() override;
  void run() override;
  void release() override;
  bool validate() override;

 private:
  HornetGraph &G_Reg, &G_Tran;
  vid_t num_components;
  vid_t* d_component_labels;
  vid_t* d_num_components;
};
}
