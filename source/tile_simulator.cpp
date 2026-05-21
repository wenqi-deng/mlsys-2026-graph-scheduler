/* TileSimulator performs tile-by-tile OOM and latency simulation, including split-k accumulators and retained tensors. */

#include "tile_simulator.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace solver {

	TileSimulator::TileSimulator(const GraphInfo &graph,
								 const ScheduleState &state,
								 const Candidate &candidate,
								 const std::vector<size_t> &external_inputs,
								 const std::vector<size_t> &boundary_outputs,
								 size_t tip_output_tensor,
								 const mlsys::Granularity &granularity,
								 const std::vector<size_t> &retained_outputs)
		: graph_(graph), state_(state), candidate_(candidate),
		  external_inputs_(external_inputs),
		  boundary_outputs_(boundary_outputs),
		  tip_output_tensor_(tip_output_tensor), granularity_(granularity),
		  retained_outputs_(retained_outputs.begin(), retained_outputs.end()),
		  candidate_ops_(candidate.ops.begin(), candidate.ops.end()),
		  boundary_output_set_(boundary_outputs.begin(),
							   boundary_outputs.end()) {
	}

	std::optional<TileSimulationResult>
	TileSimulator::simulate(const std::vector<int64_t> &spatial_order) const {
		if (granularity_.width <= 0 || granularity_.height <= 0 ||
			granularity_.depth <= 0 || candidate_.ops.empty()) {
			return std::nullopt;
		}

		const bool matmul_pointwise_chain =
			is_matmul_pointwise_chain_candidate();
		const size_t simulation_tip_op =
			matmul_pointwise_chain ? candidate_.ops.front() : candidate_.tip_op;
		const size_t simulation_tip_tensor =
			matmul_pointwise_chain
				? graph_.single_output(simulation_tip_op)
				: tip_output_tensor_;

		const auto &tip = graph_.problem().tensors.at(simulation_tip_tensor);
		const int64_t tile_rows = ceil_div(tip.height, granularity_.height);
		const int64_t tile_cols = ceil_div(tip.width, granularity_.width);
		const int64_t expected_tiles = tile_rows * tile_cols;
		if (static_cast<int64_t>(spatial_order.size()) != expected_tiles) {
			return std::nullopt;
		}
		std::vector<char> seen(static_cast<size_t>(expected_tiles), 0);
		for (int64_t flat : spatial_order) {
			if (flat < 0 || flat >= expected_tiles ||
				seen[static_cast<size_t>(flat)]) {
				return std::nullopt;
			}
			seen[static_cast<size_t>(flat)] = 1;
		}

		for (size_t t : retained_outputs_) {
			if (!boundary_output_set_.count(t)) {
				return std::nullopt;
			}
		}

		const bool sink_is_matmul = graph_.is_matmul(simulation_tip_op);
		const int64_t full_k =
			sink_is_matmul ? graph_.get_inner_dim(simulation_tip_op) : 1;
		const int64_t native_k =
			std::max<int64_t>(1, graph_.problem().native_granularity.width);
		if (sink_is_matmul && granularity_.depth > native_k) {
			return std::nullopt;
		}
		const int64_t k_stride =
			sink_is_matmul ? std::max<int64_t>(1, granularity_.depth) : 1;

		int64_t retained_full = 0;
		for (size_t t : retained_outputs_) {
			retained_full += graph_.tensor_elements(t);
		}
		if (retained_full > graph_.problem().fast_memory_capacity) {
			return std::nullopt;
		}

		std::unordered_map<size_t, std::vector<Rect>> fixed_resident;
		for (size_t t : external_inputs_) {
			if (state_.resident_tensors.at(t)) {
				fixed_resident[t].push_back(full_tensor_rect(t));
			}
		}
		if (union_area(fixed_resident) >
			graph_.problem().fast_memory_capacity) {
			return std::nullopt;
		}

		std::unordered_map<size_t, std::vector<Rect>> dynamic_cache;
		std::unordered_map<size_t, std::vector<Rect>> retained_boundary;
		std::unordered_map<size_t, std::vector<Rect>> stored_boundary;

		TileSimulationResult result;
		for (int64_t flat : spatial_order) {
			const int64_t tile_row = flat / tile_cols;
			const int64_t tile_col = flat % tile_cols;

			for (int64_t k_begin = 0; k_begin < full_k; k_begin += k_stride) {
				StepRequirements step =
					build_step_requirements(tile_row, tile_col, k_begin);

				int64_t additional_load = 0;
				std::unordered_map<size_t, std::vector<Rect>> step_footprint =
					fixed_resident;
				for (const auto &[tensor_id, rects] : retained_boundary) {
					auto &dst = step_footprint[tensor_id];
					dst.insert(dst.end(), rects.begin(), rects.end());
				}
				for (const auto &[tensor_id, rects] : step.external_inputs) {
					auto covers = fixed_resident[tensor_id];
					const auto cache_it = dynamic_cache.find(tensor_id);
					if (cache_it != dynamic_cache.end()) {
						covers.insert(covers.end(), cache_it->second.begin(),
									  cache_it->second.end());
					}
					additional_load += incremental_area(covers, rects);
					auto &dst = step_footprint[tensor_id];
					dst.insert(dst.end(), rects.begin(), rects.end());
				}
				for (const auto &[tensor_id, rects] : step.boundary_outputs) {
					auto &dst = step_footprint[tensor_id];
					dst.insert(dst.end(), rects.begin(), rects.end());
				}
				for (const auto &[tensor_id, rects] : step.accumulators) {
					auto &dst = step_footprint[tensor_id];
					dst.insert(dst.end(), rects.begin(), rects.end());
				}

				const int64_t working_set = union_area(step_footprint);
				result.peak_fast_memory =
					std::max(result.peak_fast_memory, working_set);
				if (working_set > graph_.problem().fast_memory_capacity) {
					return std::nullopt;
				}

				int64_t additional_store = 0;
				for (const auto &[tensor_id, rects] : step.boundary_outputs) {
					if (retained_outputs_.count(tensor_id)) {
						auto &dst = retained_boundary[tensor_id];
						dst.insert(dst.end(), rects.begin(), rects.end());
						continue;
					}
					additional_store +=
						incremental_area(stored_boundary[tensor_id], rects);
					auto &dst = stored_boundary[tensor_id];
					dst.insert(dst.end(), rects.begin(), rects.end());
				}

				const double mem_latency =
					graph_.transfer_time(additional_load + additional_store);
				result.latency += std::max(step.compute_latency, mem_latency);

				dynamic_cache = step.external_inputs;
			}
		}

		return result;
	}

	int64_t TileSimulator::ceil_div(int64_t a, int64_t b) const {
		return (a + b - 1) / b;
	}

	TileSimulator::Rect
	TileSimulator::full_tensor_rect(size_t tensor_id) const {
		const auto &tensor = graph_.problem().tensors.at(tensor_id);
		return Rect{0, tensor.height, 0, tensor.width};
	}

	TileSimulator::Rect TileSimulator::make_tensor_rect(size_t tensor_id,
										 int64_t tile_row,
										 int64_t tile_col) const {
		const auto &tensor = graph_.problem().tensors.at(tensor_id);
		const int64_t row_begin = tile_row * granularity_.height;
		const int64_t col_begin = tile_col * granularity_.width;
		return Rect{
			row_begin,
			std::min<int64_t>(tensor.height, row_begin + granularity_.height),
			col_begin,
			std::min<int64_t>(tensor.width, col_begin + granularity_.width),
		};
	}

	TileSimulator::Rect TileSimulator::make_tip_rect(int64_t tile_row,
													 int64_t tile_col) const {
		return make_tensor_rect(tip_output_tensor_, tile_row, tile_col);
	}

	TileSimulator::Range TileSimulator::full_k_range(size_t op_id) const {
		return Range{0, graph_.get_inner_dim(op_id)};
	}

	TileSimulator::Range TileSimulator::clamp_k_range(size_t op_id,
													  int64_t k_begin) const {
		const int64_t full_k = graph_.get_inner_dim(op_id);
		const int64_t size = std::min<int64_t>(
			std::max<int64_t>(1, granularity_.depth), full_k - k_begin);
		return Range{k_begin, size};
	}

	TileSimulator::MatMulLayout
	TileSimulator::detect_matmul_layout(size_t op_id) const {
		const auto &op = graph_.problem().ops.at(op_id);
		if (!graph_.is_matmul(op_id) || op.inputs.size() != 2) {
			throw std::runtime_error(
				"MatMul layout requested for non-matmul op");
		}
		const auto &lhs = graph_.problem().tensors.at(op.inputs[0]);
		const auto &rhs = graph_.problem().tensors.at(op.inputs[1]);
		const auto &out =
			graph_.problem().tensors.at(graph_.single_output(op_id));

		if (lhs.width == rhs.height && out.height == lhs.height &&
			out.width == rhs.width) {
			return MatMulLayout::Standard;
		}
		if (lhs.height == rhs.width && out.height == lhs.width &&
			out.width == rhs.height) {
			return MatMulLayout::Swapped;
		}
		if (lhs.width == rhs.height) {
			return MatMulLayout::Standard;
		}
		if (lhs.height == rhs.width) {
			return MatMulLayout::Swapped;
		}
		throw std::runtime_error("MatMul dimension mismatch");
	}

	bool TileSimulator::is_matmul_pointwise_chain_candidate() const {
		if (candidate_.kind != CandidateType::MatMulPointwiseChain ||
			candidate_.ops.empty() || !graph_.is_matmul(candidate_.ops.front())) {
			return false;
		}
		for (size_t i = 1; i < candidate_.ops.size(); ++i) {
			if (!graph_.is_pointwise(candidate_.ops[i])) {
				return false;
			}
		}
		return true;
	}

	double TileSimulator::pointwise_chain_base_cost_per_tile() const {
		double cost = 0.0;
		for (size_t i = 1; i < candidate_.ops.size(); ++i) {
			cost += static_cast<double>(
				graph_.problem().ops.at(candidate_.ops[i]).base_cost);
		}
		return cost;
	}

	TileSimulator::Rect
	TileSimulator::make_lhs_rect(size_t op_id, const Rect &out_rect,
								 const Range &k_range) const {
		const MatMulLayout layout = detect_matmul_layout(op_id);
		if (layout == MatMulLayout::Standard) {
			return Rect{
				out_rect.row_begin,
				out_rect.row_end,
				k_range.begin,
				k_range.begin + k_range.size,
			};
		}
		return Rect{
			k_range.begin,
			k_range.begin + k_range.size,
			out_rect.row_begin,
			out_rect.row_end,
		};
	}

	TileSimulator::Rect
	TileSimulator::make_rhs_rect(size_t op_id, const Rect &out_rect,
								 const Range &k_range) const {
		const MatMulLayout layout = detect_matmul_layout(op_id);
		if (layout == MatMulLayout::Standard) {
			return Rect{
				k_range.begin,
				k_range.begin + k_range.size,
				out_rect.col_begin,
				out_rect.col_end,
			};
		}
		return Rect{
			out_rect.col_begin,
			out_rect.col_end,
			k_range.begin,
			k_range.begin + k_range.size,
		};
	}

	TileSimulator::SliceKind
	TileSimulator::slice_kind_for_consumer_input(size_t op_id,
												 int input_index) const {
		if (!graph_.is_matmul(op_id)) {
			return SliceKind::Spatial;
		}
		const MatMulLayout layout = detect_matmul_layout(op_id);
		if (layout == MatMulLayout::Standard) {
			return (input_index == 0) ? SliceKind::ReductionOnCols
									  : SliceKind::ReductionOnRows;
		}
		return (input_index == 0) ? SliceKind::ReductionOnRows
								  : SliceKind::ReductionOnCols;
	}

	double TileSimulator::op_compute_latency(
		size_t op_id, const Rect &out_rect,
		const std::optional<Range> &reduction_range,
		SliceKind slice_kind) const {
		const auto &op = graph_.problem().ops.at(op_id);
		if (graph_.is_pointwise(op_id)) {
			return static_cast<double>(op.base_cost);
		}

		const int64_t native_k =
			std::max<int64_t>(1, graph_.problem().native_granularity.width);
		double factor = static_cast<double>(graph_.get_inner_dim(op_id)) /
						static_cast<double>(native_k);
		if (reduction_range.has_value()) {
			factor = static_cast<double>(reduction_range->size) /
					 static_cast<double>(native_k);
		}

		if (slice_kind == SliceKind::ReductionOnCols) {
			factor *=
				static_cast<double>(out_rect.col_end - out_rect.col_begin) /
				static_cast<double>(native_k);
		} else if (slice_kind == SliceKind::ReductionOnRows) {
			factor *=
				static_cast<double>(out_rect.row_end - out_rect.row_begin) /
				static_cast<double>(native_k);
		}

		return static_cast<double>(op.base_cost) * factor;
	}

	void TileSimulator::add_rect(
		std::unordered_map<size_t, std::vector<Rect>> *rects, size_t tensor_id,
		const Rect &rect) const {
		if (rect.row_begin >= rect.row_end || rect.col_begin >= rect.col_end) {
			return;
		}
		auto &dst = (*rects)[tensor_id];
		dst.push_back(rect);
	}

	int64_t TileSimulator::union_area(const std::vector<Rect> &rects) const {
		if (rects.empty()) {
			return 0;
		}

		std::vector<int64_t> rows;
		std::vector<int64_t> cols;
		rows.reserve(rects.size() * 2);
		cols.reserve(rects.size() * 2);
		for (const Rect &rect : rects) {
			if (rect.row_begin >= rect.row_end ||
				rect.col_begin >= rect.col_end) {
				continue;
			}
			rows.push_back(rect.row_begin);
			rows.push_back(rect.row_end);
			cols.push_back(rect.col_begin);
			cols.push_back(rect.col_end);
		}
		if (rows.empty() || cols.empty()) {
			return 0;
		}
		std::sort(rows.begin(), rows.end());
		rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
		std::sort(cols.begin(), cols.end());
		cols.erase(std::unique(cols.begin(), cols.end()), cols.end());

		int64_t area = 0;
		for (size_t r = 0; r + 1 < rows.size(); ++r) {
			for (size_t c = 0; c + 1 < cols.size(); ++c) {
				const int64_t row0 = rows[r];
				const int64_t row1 = rows[r + 1];
				const int64_t col0 = cols[c];
				const int64_t col1 = cols[c + 1];
				bool covered = false;
				for (const Rect &rect : rects) {
					if (rect.row_begin <= row0 && rect.row_end >= row1 &&
						rect.col_begin <= col0 && rect.col_end >= col1) {
						covered = true;
						break;
					}
				}
				if (covered) {
					area += (row1 - row0) * (col1 - col0);
				}
			}
		}
		return area;
	}

	int64_t TileSimulator::union_area(
		const std::unordered_map<size_t, std::vector<Rect>> &rects) const {
		int64_t total = 0;
		for (const auto &[tensor_id, tensor_rects] : rects) {
			(void)tensor_id;
			total += union_area(tensor_rects);
		}
		return total;
	}

	int64_t
	TileSimulator::incremental_area(const std::vector<Rect> &existing,
									const std::vector<Rect> &added) const {
		if (added.empty()) {
			return 0;
		}
		std::vector<Rect> all = existing;
		all.insert(all.end(), added.begin(), added.end());
		return union_area(all) - union_area(existing);
	}

	std::string TileSimulator::request_key(
		size_t tensor_id, const Rect &rect,
		const std::optional<Range> &reduction_range) const {
		std::string key =
			std::to_string(tensor_id) + ":" + std::to_string(rect.row_begin) +
			"," + std::to_string(rect.row_end) + ":" +
			std::to_string(rect.col_begin) + "," + std::to_string(rect.col_end);
		if (reduction_range.has_value()) {
			key += ":" + std::to_string(reduction_range->begin) + "," +
				   std::to_string(reduction_range->size);
		}
		return key;
	}

	bool TileSimulator::is_split_k_accumulator(
		size_t tensor_id, const std::optional<Range> &reduction_range) const {
		if (!reduction_range.has_value()) {
			return false;
		}
		const int producer = graph_.producer_of_tensor().at(tensor_id);
		if (producer == -1 ||
			!candidate_ops_.count(static_cast<size_t>(producer)) ||
			!graph_.is_matmul(static_cast<size_t>(producer))) {
			return false;
		}
		return reduction_range->size <
			   graph_.get_inner_dim(static_cast<size_t>(producer));
	}

	void TileSimulator::collect_tensor_requirements(
		size_t tensor_id, const Rect &rect,
		const std::optional<Range> &reduction_range, SliceKind slice_kind,
		bool boundary_ready, StepRequirements *req,
		std::unordered_set<std::string> *visited) const {
		if (rect.row_begin >= rect.row_end || rect.col_begin >= rect.col_end) {
			return;
		}
		if (boundary_ready && boundary_output_set_.count(tensor_id)) {
			add_rect(&req->boundary_outputs, tensor_id, rect);
		}
		if (is_split_k_accumulator(tensor_id, reduction_range)) {
			add_rect(&req->accumulators, tensor_id, rect);
		}

		const std::string key = request_key(tensor_id, rect, reduction_range);
		if (!visited->insert(key).second) {
			return;
		}

		const int producer = graph_.producer_of_tensor().at(tensor_id);
		if (producer == -1 ||
			!candidate_ops_.count(static_cast<size_t>(producer))) {
			add_rect(&req->external_inputs, tensor_id, rect);
			return;
		}

		const size_t op_id = static_cast<size_t>(producer);
		req->compute_latency +=
			op_compute_latency(op_id, rect, reduction_range, slice_kind);

		const auto &op = graph_.problem().ops.at(op_id);
		if (graph_.is_pointwise(op_id)) {
			for (size_t input_tensor : op.inputs) {
				collect_tensor_requirements(input_tensor, rect, std::nullopt,
											slice_kind, true, req, visited);
			}
			return;
		}

		const Range used_k = reduction_range.value_or(full_k_range(op_id));
		const Rect lhs_rect = make_lhs_rect(op_id, rect, used_k);
		const Rect rhs_rect = make_rhs_rect(op_id, rect, used_k);
		collect_tensor_requirements(op.inputs[0], lhs_rect, std::nullopt,
									slice_kind_for_consumer_input(op_id, 0),
									true, req, visited);
		collect_tensor_requirements(op.inputs[1], rhs_rect, std::nullopt,
									slice_kind_for_consumer_input(op_id, 1),
									true, req, visited);
	}

	TileSimulator::StepRequirements
	TileSimulator::build_step_requirements(int64_t tile_row, int64_t tile_col,
										   int64_t k_begin) const {
		StepRequirements req;
		std::unordered_set<std::string> visited;
		const bool matmul_pointwise_chain =
			is_matmul_pointwise_chain_candidate();
		const size_t matmul_root =
			matmul_pointwise_chain ? candidate_.ops.front() : candidate_.tip_op;
		const size_t root_output_tensor =
			matmul_pointwise_chain ? graph_.single_output(matmul_root)
									   : tip_output_tensor_;
		const Rect root_rect = make_tensor_rect(root_output_tensor, tile_row,
											 tile_col);

		if (graph_.is_matmul(matmul_root)) {
			const Range k_range = clamp_k_range(matmul_root, k_begin);
			const int64_t full_k = graph_.get_inner_dim(matmul_root);
			const bool tip_boundary_ready =
				(k_range.begin + k_range.size >= full_k);
			collect_tensor_requirements(root_output_tensor, root_rect, k_range,
										SliceKind::Spatial, tip_boundary_ready,
										&req, &visited);
			if (matmul_pointwise_chain && k_begin == 0) {
				req.compute_latency += pointwise_chain_base_cost_per_tile();
			}
			if (matmul_pointwise_chain && tip_boundary_ready) {
				for (size_t i = 1; i < candidate_.ops.size(); ++i) {
					const auto &pw = graph_.problem().ops.at(candidate_.ops[i]);
					for (size_t input_tensor : pw.inputs) {
						const int producer =
							graph_.producer_of_tensor().at(input_tensor);
						if (producer != -1 &&
							candidate_ops_.count(static_cast<size_t>(producer))) {
							continue;
						}
						add_rect(&req.external_inputs, input_tensor,
								 make_tensor_rect(input_tensor, tile_row, tile_col));
					}
				}
				add_rect(&req.boundary_outputs, tip_output_tensor_,
						 make_tip_rect(tile_row, tile_col));
			}
		} else {
			collect_tensor_requirements(tip_output_tensor_, root_rect,
										std::nullopt, SliceKind::Spatial, true,
										&req, &visited);
		}
		return req;
	}

} // namespace solver
