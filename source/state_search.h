#ifndef MLSYS_SOLVER_STATE_SEARCH_H_
#define MLSYS_SOLVER_STATE_SEARCH_H_

#include "candidate_generator.h"
#include "graph_utils.h"
#include "search_types.h"

namespace solver {

class BeamSearchScheduler {
  public:
	BeamSearchScheduler(const GraphInfo &graph, const SearchConfig &config)
		: graph_(graph), config_(config), generator_(graph, config) {
	}

	mlsys::Solution build_solution() const;

  private:
	double estimated_total_latency(const ScheduleState &state) const;
	bool is_all_outputs_stored(const ScheduleState &state) const;

	const GraphInfo &graph_;
	SearchConfig config_;
	CandidateGenerator generator_;
};

} // namespace solver

#endif
