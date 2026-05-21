#ifndef SEARCH_TYPES_H_
#define SEARCH_TYPES_H_

#include "mlsys.h"
#include <vector>

namespace solver {

enum class CandidateType {
	SingleOp,
	LinearChain,
	MatMulPointwiseChain,
	BackwardCone,
	RecomputeCone,
};

struct Candidate {
	CandidateType kind = CandidateType::SingleOp;
	size_t tip_op = 0;
	std::vector<size_t> ops; // topologically sorted
	bool uses_recompute = false;
};

struct SearchConfig {
	int beam_width = 8;
	int max_candidate_ops = 6;
	int max_chain_len = 16;
	int max_recompute_depth = 2;
	int max_retained_outputs = 2;
	int max_search_steps = 512;
	size_t max_candidate_num = 48;
};

/*
 * Field guide:
 * covered_ops: operations already covered by the partial schedule.
 * stored_tensors: tensors available from slow memory.
 * resident_tensors: tensors retained in fast memory across exactly one boundary.
 * schedule: ordered accepted subgraphs on this search path.
 */
struct ScheduleState {
	std::vector<bool> covered_ops;
	// ops that are covered by the current schedule
	std::vector<bool> stored_tensors;
	// stored in slow memory
	std::vector<bool> resident_tensors;
	// resident in fast memory from previous step
	int covered_count = 0;
	double total_latency = 0.0;

	std::vector<mlsys::Subgraph> schedule;
	// Schduled subgraphs
};

} // namespace solver

#endif
