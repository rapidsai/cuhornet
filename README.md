# Hornet #

NOTE: The cuhornet repository is a copy of https://github.com/hornet-gt/hornet that is being maintained by the RAPIDS
cugraph team while we use it in our library.  We currently only use headers to provide the ktruss implementation.

This library does not currently build.  Since we only use headers, we are not maintaining the build processes for
this library.  We expect to drop support for this entirely in early 2024.

This repository provides the Hornet data structure and algorithms on sparse graphs and matrices.

## Getting Started ##

The document is organized as follows:

* [Requirements](#requirements)
* [Quick start](#quick-start)
* [Supported graph formats](#supported-graph-formats)
* [Code Documentation](#code-documentation)
* [Notes](#notes)
* [Reporting bugs and contributing](#reporting-bugs-and-contributing)
* [Publications](#publications)
* [Hornet Developers](#hornet-developers)
* [License](#license)

### Requirements ###

* [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) 9 or greater.
* GCC or [Clang](https://clang.llvm.org) host compiler with support for C++14.
  Note, the compiler must be compatible with the related CUDA toolkit version.
  For more information see [CUDA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
* [CMake](https://cmake.org) v3.8 or greater.
* 64-bit Operating System (Ubuntu 16.04 or above suggested).

### Quick start ###

The following basic steps are required to build and execute Hornet:
```bash
git clone --recursive https://github.com/hornet-gt/hornet
export CUDACXX=<path_to_CUDA_nvcc_compiler>
cd hornet/build
cmake ..
make -j
```

To build HornetsNest (algorithms directory):
```bash
cd hornetsnest/build
cmake ..
make -j
```

By default, the CUDA compiler `nvcc` uses `gcc/g++` found in the current
execution search path as host compiler
(`cc --version` to get the default compiler on the actual system).
To force a different host compiler for compiling C++ files (`*.cpp`)
you need to set the following environment variables:
 ```bash
CC=<path_to_host_C_compiler>
CXX=<path_to_host_C++_compiler>
```

Note: host `.cpp` compiler and host side `.cu` compiler may be different.
The host side compiler must be compatible with the current CUDA Toolkit
version installed on the system
(see [CUDA Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)).

The syntax and the input parameters of Hornet are explained in detail in
 `docs/Syntax.txt`. They can also be found by typing `./HornetTest --help`.

### Supported graph formats ###

Hornet supports the following graph input formats:

* Market (.mtx), [The University of Florida Sparse Matrix Collection](http://www.cise.ufl.edu/research/sparse/matrices/)
* Metis (.graph), [10th DIMACS Implementation Challenge](http://www.cc.gatech.edu/dimacs10/)
* SNAP (.txt), [Stanford Network Analysis Project](http://snap.stanford.edu/)
* Dimacs9th (.gr), [9th DIMACS Implementation Challenge](http://www.dis.uniroma1.it/challenge9/)
* The Koblenz Network Collection (out.< name >), [The Koblenz Network Collection](http://konect.uni-koblenz.de/)
* Network Data Repository (.edges), [Network Data Repository](http://networkrepository.com/index.php)
* Binary (.bin)

Hornet directly deduces the graph structure (directed/undirected) from the input file header.

Hornet allows reading the input graph by using a fixed binary format to speed up the file loading.
The binary file is generated by Hornet with the `--binary` command line option.

### Code Documentation ###

The code documentation is located in the `docs` directory (*doxygen* html format).

### Notes ###

* Hornet has been checked with the following tools to ensure the code quality:
    * [clang++ v4.0: warnings](https://clang.llvm.org/docs/DiagnosticsReference.html)
    * [clang-tidy](http://clang.llvm.org/extra/clang-tidy/): warnings and code styles
* Hornet has been tested with the following tools: (see [`CodeCheck`](docs/CodeCheck.md))
    * [cuda-memcheck](http://docs.nvidia.com/cuda/cuda-memcheck/)
    * [valgrind v3.13](http://valgrind.org/)
    * [clang static analyzer v279](https://clang-analyzer.llvm.org/)

### Reporting bugs and contributing ###

If you find any bugs please report them by using the repository (github **issues** panel).
We are also ready to engage in improving and extending the framework if you request new features.

## Hornet Algorithms ##

|           Algorithm                 |    Static     | Dynamic  |
| :-----------------------------------|:-------------:|:--------:|
| (BFS) Breadth-first Search          |     yes       | on-going |
| (SSSP) Single-Source Shortest Path  |     yes       | on-going |
| (CC) Connected Components           |     yes       | on-going |
| (SCC) Strongly Connected Components |    to-do      |  to-do   |
| (MST) Minimum Spanning Tree         |   on-going    |  to-do   |
| (BC) Betweeness Centrality          |     yes       | on-going |
| (PG) Page Rank                      |     yes       |   yes    |
| (TC) Triangle Counting              |     yes       | on-going |
| (KC) Katz Centrality                |     yes       |   yes    |
| (MIS) Maximal Independent Set       |   on-going    |  to-do   |
| (MF) Maximum Flow                   |    to-do      |  to-do   |
| (CC) Clustering Coeffient           |     yes       |  to-do   |
| (ST) St-Connectivity                |    to-do      |  to-do   |
| (TC) Transitive Closure             |    to-do      |  to-do   |
| Community Detection                 |    on-going   |  to-do   |
| Temporal Motif Finding              |   on-going    |  to-do   |
| Sparse Vector-Matrix Multiplication |     yes       |  to-do   |
| Jaccard indices                     |   on-going    |  to-do   |
| Energy/Parity Game                  |   on-going    |  to-do   |

## Publications ##

* F. Busato, O. Green, N. Bombieri, D. Bader, **“Hornet: An Efficient Data Structure for Dynamic Sparse Graphs and Matrices”**, IEEE High Performance Extreme Computing Conference (HPEC), Waltham, Massachusetts, 2018
[link](https://www.researchgate.net/publication/327569751_Hornet_An_Efficient_Data_Structure_for_Dynamic_Sparse_Graphs_and_Matrices_on_GPUs) 
* Oded Green, David A. Bader, **"cuSTINGER: Supporting dynamic graph algorithms
  for GPUs"**,
  IEEE High Performance Extreme Computing Conference (HPEC), 13-15 September,
  2016, Waltham, MA, USA, pp. 1-6.
  [link](https://www.researchgate.net/publication/308174457_cuSTINGER_Supporting_dynamic_graph_algorithms_for_GPUs)
* Fox, O. Green, K. Gabert, X. An, D. Bader, **“Fast and Adaptive List Intersections on the GPU”**, IEEE High Performance Extreme Computing Conference (HPEC), Waltham, Massachusetts, 2018 \**HPEC Graph Challenge Finalist *\*
* O. Green, J. Fox, A. Tripathy, A. Watkins, K. Gabert, E. Kim, X. An, K. Aatish, D. Bader, **“Logarithmic Radix Binning and Vectorized Triangle Counting”**, IEEE High Performance Extreme Computing Conference (HPEC), Waltham, Massachusetts, 2018 (HPEC Graph Challenge Innovation Award)
* A. van der Grinten, E. Bergamini, O. Green, H. Meyerhenke, D. Bader, **“Scalable Katz Ranking Computation in Large Dynamic Graphs”**, European Symposium on Algorithms, Helsinki, Finland, 2018 
* Oded Green, James Fox, Euna Kim, Federico Busato, Nicola Bombieri,
  Kartik Lakhotia, Shijie Zhou, Shreyas Singapura, Hanqing Zeng,
  Rajgopal Kannan, Viktor Prasanna, David A. Bader,
  **"Quickly Finding a Truss in a Haystack"**,
  IEEE/Amazon/DARPA Graph Challenge, \**Innovation Awards*\*.
* Devavret Makkar, David A. Bader, Oded Green,
  **Exact and Parallel Triangle Counting in Streaming Graphs**,
  IEEE Conference on High Performance Computing, Data, and Analytics (HiPC),
  18-21 December 2017, Jaipur, India, pp. 1-10.
* [A. Tripathy](http://www.aloktripathy.me), F. Hohman, D.H Chau, O. Green,
  **"Scalable K-Core Decomposition for Static Graphs Using a Dynamic Graph Data Structure"**,
  IEEE International Conference on Big Data,
  Seattle, Washington, 2018
  [link](https://www.researchgate.net/publication/328874544_Scalable_K-Core_Decomposition_for_Static_Graphs_Using_a_Dynamic_Graph_Data_Structure)
---
### <center>If you find this software useful in academic work, please acknowledge Hornet. </center> ###
***

## Hornet Developers ##

* `Federico Busato`, Ph.D. Student, University of Verona (Italy)
* `Oded Green`, Researcher, Georgia Institute of Technology
* `Federico Busato`, Ph.D. Student, University of Verona (Italy)
* `Oded Green`, Researcher, Georgia Institute of Technology
* `James Fox`, Ph.D. Student, Georgia Institute of Technology : *Maximal Independent Set*, *Temporal Motif Finding*
* `Devavret Makkar`, Ph.D. Student, Georgia Institute of Technology : *Triangle Counting*
* `Elisabetta Bergamini`, Ph.D. Student, Karlsruhe Institute of Technology (Germany) : *Katz Centrality*
* `Euna Kim`, Ph.D. Student, Georgia Institute of Technology : *Dynamic PageRank*
* ...

## License ##

> BSD 3-Clause License
>
> Copyright (c) 2017, Hornet
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>   list of conditions and the following disclaimer.
>
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
>
> * Neither the name of the copyright holder nor the names of its
>   contributors may be used to endorse or promote products derived from
>   this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
