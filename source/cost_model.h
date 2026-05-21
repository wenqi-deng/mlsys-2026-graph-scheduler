#ifndef COST_MODEL_H_
#define COST_MODEL_H_

#include "graph_utils.h"
#include "search_types.h"
#include <optional>
#include <utility>
#include <vector>

namespace solver {

struct CandidateEvaluation {
	mlsys::Subgraph subgraph;
	std::vector<size_t> boundary_outputs;
	size_t newly_covered_ops = 0;
};

class Evaluator {
  public:
	Evaluator(const GraphInfo &graph, const SearchConfig &config,
			  const ScheduleState &state, const Candidate &candidate);
	std::optional<CandidateEvaluation> evaluate_candidate();

  private:
	struct CandidateSummary {
		std::vector<size_t> boundary_outputs;
		std::vector<size_t> external_inputs;
		size_t tip_output_tensor = 0;
		bool linear = false;
		bool single_matmul = false;
		bool matmul_pointwise_chain = false;
		bool has_matmul = false;
		bool all_pointwise = false;
	};

	static int64_t ceil_div(int64_t a, int64_t b);

	std::vector<int64_t> make_dim_choices(int64_t dim, int64_t native_dim);
	std::vector<int64_t> make_k_choices();
	std::vector<std::pair<int64_t, int64_t>>
	make_spatial_choices(const std::vector<int64_t> &w_choices,
						 const std::vector<int64_t> &h_choices) const;

	void summarize_candidate();
	std::optional<CandidateEvaluation>
	evaluate_at_granularity(const mlsys::Granularity &g);
	std::optional<CandidateEvaluation>
	evaluate_single_matmul_candidate(
		const std::vector<std::pair<int64_t, int64_t>> &spatial_choices);
	std::optional<CandidateEvaluation>
	evaluate_matmul_pointwise_chain_candidate(
		const std::vector<std::pair<int64_t, int64_t>> &spatial_choices);
	std::optional<CandidateEvaluation>
	evaluate_pointwise_candidate(
		const std::vector<std::pair<int64_t, int64_t>> &spatial_choices);

	std::vector<int64_t> build_raster_order(int64_t rows, int64_t cols);
	std::vector<int64_t> build_snake_order(int64_t rows, int64_t cols);

	CandidateEvaluation make_evaluation(
		const mlsys::Granularity &g,
		const std::vector<size_t> &retained_tensors,
		const std::optional<mlsys::TraversalOrder> &traversal,
		double latency) const;

	const GraphInfo &graph_;
	const SearchConfig &config_;
	const ScheduleState &state_;
	const Candidate &candidate_;
	CandidateSummary summary_;
};

} // namespace solver

#endif
