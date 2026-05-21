#ifndef TILE_SIMULATOR_H_
#define TILE_SIMULATOR_H_

#include "graph_utils.h"
#include "search_types.h"
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace solver {

	struct TileSimulationResult {
		double latency = 0.0;
		int64_t peak_fast_memory = 0;
	};

	class TileSimulator {
	public:
		TileSimulator(const GraphInfo &graph, const ScheduleState &state,
					  const Candidate &candidate,
					  const std::vector<size_t> &external_inputs,
					  const std::vector<size_t> &boundary_outputs,
					  size_t tip_output_tensor,
					  const mlsys::Granularity &granularity,
					  const std::vector<size_t> &retained_outputs);

		std::optional<TileSimulationResult>
		simulate(const std::vector<int64_t> &spatial_order) const;

	private:
		struct Rect {
			int64_t row_begin = 0;
			int64_t row_end = 0;
			int64_t col_begin = 0;
			int64_t col_end = 0;
		};

		struct Range {
			int64_t begin = 0;
			int64_t size = 0;
		};

		enum class SliceKind {
			Spatial,
			ReductionOnRows,
			ReductionOnCols,
		};

		struct StepRequirements {
			std::unordered_map<size_t, std::vector<Rect>> external_inputs;
			std::unordered_map<size_t, std::vector<Rect>> boundary_outputs;
			std::unordered_map<size_t, std::vector<Rect>> accumulators;
			double compute_latency = 0.0;
		};

		enum class MatMulLayout {
			Standard,
			Swapped,
		};

		int64_t ceil_div(int64_t a, int64_t b) const;
		Rect full_tensor_rect(size_t tensor_id) const;
		Rect make_tensor_rect(size_t tensor_id, int64_t tile_row,
							 int64_t tile_col) const;
		Rect make_tip_rect(int64_t tile_row, int64_t tile_col) const;
		Range full_k_range(size_t op_id) const;
		Range clamp_k_range(size_t op_id, int64_t k_begin) const;
		MatMulLayout detect_matmul_layout(size_t op_id) const;
		bool is_matmul_pointwise_chain_candidate() const;
		double pointwise_chain_base_cost_per_tile() const;

		Rect make_lhs_rect(size_t op_id, const Rect &out_rect,
						   const Range &k_range) const;
		Rect make_rhs_rect(size_t op_id, const Rect &out_rect,
						   const Range &k_range) const;
		SliceKind slice_kind_for_consumer_input(size_t op_id,
												int input_index) const;

		double op_compute_latency(size_t op_id, const Rect &out_rect,
								  const std::optional<Range> &reduction_range,
								  SliceKind slice_kind) const;

		void add_rect(std::unordered_map<size_t, std::vector<Rect>> *rects,
					  size_t tensor_id, const Rect &rect) const;
		int64_t union_area(const std::vector<Rect> &rects) const;
		int64_t union_area(
			const std::unordered_map<size_t, std::vector<Rect>> &rects) const;
		int64_t incremental_area(const std::vector<Rect> &existing,
								 const std::vector<Rect> &added) const;

		std::string
		request_key(size_t tensor_id, const Rect &rect,
					const std::optional<Range> &reduction_range) const;
		void collect_tensor_requirements(
			size_t tensor_id, const Rect &rect,
			const std::optional<Range> &reduction_range, SliceKind slice_kind,
			bool boundary_ready, StepRequirements *req,
			std::unordered_set<std::string> *visited) const;
		bool is_split_k_accumulator(
			size_t tensor_id,
			const std::optional<Range> &reduction_range) const;
		StepRequirements build_step_requirements(int64_t tile_row,
												 int64_t tile_col,
												 int64_t k_begin) const;

		const GraphInfo &graph_;
		const ScheduleState &state_;
		const Candidate &candidate_;
		std::vector<size_t> external_inputs_;
		std::vector<size_t> boundary_outputs_;
		size_t tip_output_tensor_;
		mlsys::Granularity granularity_;
		std::unordered_set<size_t> retained_outputs_;
		std::unordered_set<size_t> candidate_ops_;
		std::unordered_set<size_t> boundary_output_set_;
	};

} // namespace solver

#endif
