/**
 * @brief
 * @author Oded Green                                                       <br>
 *   Georgia Institute of Technology, Computational Science and Engineering <br>                   <br>
 *   ogreen@gatech.edu
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
 * @file
 */
#include "Static/PageRank/PageRank.cuh"
#include "PageRankOperators.cuh"

namespace hornets_nest {


StaticPageRank::StaticPageRank(HornetGraph& hornet,
	            	int  iteration_max,
	            	pr_t     threshold,
	            	pr_t          damp,
		        	bool isUndirected):
                                    StaticAlgorithm(hornet),
                                    load_balancing(hornet) {
#ifdef DEBUG
	if(isUndirected==true)
		printf("Init is true\n");
	else
		printf("Init is false\n");
#endif

    setInputParameters(iteration_max, threshold, damp,isUndirected);
	hd_prdata().nV = hornet.nV();
	pool.allocate(&hd_prdata().prev_pr,  hornet.nV() + 1);
	pool.allocate(&hd_prdata().curr_pr,  hornet.nV() + 1);
	pool.allocate(&hd_prdata().abs_diff, hornet.nV() + 1);
	pool.allocate(&hd_prdata().contri,   hornet.nV() + 1);
	pool.allocate(&hd_prdata().reduction_out, 1);

	reset();
}

StaticPageRank::~StaticPageRank() {
    release();
}

void StaticPageRank::release() {
  hd_prdata().prev_pr = nullptr;
  hd_prdata().curr_pr = nullptr;
  hd_prdata().abs_diff = nullptr;
  hd_prdata().contri = nullptr;
  hd_prdata().reduction_out = nullptr;
  host::free(host_page_rank);
}

void StaticPageRank::reset(){
	hd_prdata().iteration = 0;
}

void StaticPageRank::setInputParameters(int  iteration_max,
                                        pr_t threshold,
                                        pr_t damp,
                                        bool isUndirected) {
	hd_prdata().iteration_max   = iteration_max;
	hd_prdata().threshold       = threshold;
	hd_prdata().damp            = damp;
	hd_prdata().normalized_damp = (1.0f - hd_prdata().damp) /
                                  static_cast<float>(hornet.nV());
	this->isUndirected = isUndirected;
#ifdef DEBUG
	if(this->isUndirected==true)
		printf("Init is true\n");
	else
		printf("Init is false\n");
#endif
}

void StaticPageRank::run() {
	forAllnumV(hornet, InitOperator { hd_prdata });
	hd_prdata().iteration = 0;

	pr_t h_out = hd_prdata().threshold + 1;

#ifdef DEBUG
	if(this->isUndirected==true)
		printf("Run is true\n");
	else
		printf("Run is false\n");
#endif

	while(hd_prdata().iteration < hd_prdata().iteration_max &&
          h_out > hd_prdata().threshold) {

		forAllnumV(hornet, ResetCurr { hd_prdata });
		forAllVertices(hornet, ComputeContribuitionPerVertex { hd_prdata });
		if (isUndirected == true){
			forAllEdges(hornet, AddContribuitionsPush { hd_prdata }, load_balancing);
		}else{
			forAllEdges(hornet, AddContribuitionsPull { hd_prdata },load_balancing);
		}
		forAllnumV(hornet, DampAndDiffAndCopy { hd_prdata });

		forAllnumV(hornet, Sum { hd_prdata });
		hd_prdata.sync();

        host::copyFromDevice(hd_prdata().reduction_out, h_out);
		hd_prdata().iteration++;
	}
}


void StaticPageRank::printRankings() {
    pr_t*  d_scores, *h_scores;
    vid_t* d_ids, *h_ids;
    pool.allocate(&d_scores,  hornet.nV());
    pool.allocate(&d_ids,     hornet.nV());
    host::allocate(h_scores, hornet.nV());
    host::allocate(h_ids,    hornet.nV());

    gpu::copyToDevice(hd_prdata().curr_pr, hornet.nV(), d_scores);
	forAllnumV(hornet, SetIds { d_ids });

    host::copyFromDevice(d_scores, hornet.nV(), h_scores);
    host::copyFromDevice(d_ids,    hornet.nV(), h_ids);

	for (int i = 0; i < 10; i++)
        std::cout << "Pr[" << h_ids[i] << "]:= " <<  h_scores[i] << "\n";
    std::cout << std::endl;

	forAllnumV(hornet, ResetCurr { hd_prdata });
	forAllnumV(hornet, SumPr     { hd_prdata });

	pr_t h_out;
    host::copyFromDevice(hd_prdata().reduction_out, h_out);
	std::cout << "              " << std::setprecision(9) << h_out << std::endl;

	host::free(h_scores);
	host::free(h_ids);
}

const pr_t* StaticPageRank::get_page_rank_score_host() {
    host::allocate(host_page_rank, hornet.nV());
    host::copyFromDevice(hd_prdata().curr_pr, hornet.nV(), host_page_rank);
    return host_page_rank;
}

int StaticPageRank::get_iteration_count() {
	return hd_prdata().iteration;
}

bool StaticPageRank::validate() {
	return true;//TODO : Add validation code
}

PrData StaticPageRank::pr_data(void) {
    return hd_prdata;
}

}// hornets_nest namespace
