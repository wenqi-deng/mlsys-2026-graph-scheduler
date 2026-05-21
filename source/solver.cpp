#include "solver.h"
#include "graph_utils.h"
#include "state_search.h"

namespace solver {

mlsys::Solution Solver::solve(const mlsys::Problem &problem) const {
	SearchConfig cfg = config_;

	// Large graphs adjustment: keeping the search bounded >> using a wider
	// beam.
	if (problem.ops.size() > 80) {
		cfg.beam_width = 5;
		cfg.max_candidate_ops = 5;
		cfg.max_recompute_depth = 1;
		cfg.max_retained_outputs = 1;
	} else if (problem.ops.size() > 50) {
		cfg.beam_width = 4;
		cfg.max_candidate_ops = 5;
	}

	GraphInfo graph(problem);
	BeamSearchScheduler scheduler(graph, cfg);
	return scheduler.build_solution();
}

} // namespace solver
