#ifndef CANDIDATE_GENERATOR_H_
#define CANDIDATE_GENERATOR_H_

#include "graph_utils.h"
#include "search_types.h"
#include <vector>

namespace solver {

class CandidateGenerator {
  public:
	CandidateGenerator(const GraphInfo &graph, const SearchConfig &config)
		: graph_(graph), config_(config) {
	}

	std::vector<Candidate> generate(const ScheduleState &state) const;

  private:
	bool tensor_available(const ScheduleState &state, size_t tensor_id) const;
	bool is_external_tensors_available(const ScheduleState &state,
								   const std::vector<size_t> &ops) const;
	bool is_linear_extendable(const ScheduleState &state, const Candidate &cur,
						   size_t next_op) const;

	bool expand_backward_cone(const ScheduleState &state, size_t op_id,
							  int *budget, int depth, bool allow_recompute,
							  std::vector<bool> *in_candidate,
							  bool *used_recompute) const;

	std::vector<Candidate>
	generate_linear_candidates(const ScheduleState &state) const;
	std::vector<Candidate>
	generate_backward_cone_candidates(const ScheduleState &state,
									   bool allow_recompute) const;

	const GraphInfo &graph_;
	const SearchConfig &config_;
};

} // namespace solver

#endif
