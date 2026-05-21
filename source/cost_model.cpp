/* Evaluator converts a structural Candidate into an executable Subgraph by enumerating granularity/retention and invoking TileSimulator. */

/* Evaluate candidates with the tile simulator.
 *
 * The search intentionally keeps the optimization scope conservative:
 *   1. single MatMul: enumerate spatial (w,h), choose the largest feasible
 *      native-bounded k for that spatial shape, and use snake traversal;
 *   2. pointwise chains/cones: k=1 and legality is decided by tiled working set;
 *   3. retain choices are evaluated by the same simulator so OOM is never
 *      accepted just to save memory traffic.
 */

#include "cost_model.h"
#include "retention_policy.h"
#include "tile_simulator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_set>
#include <vector>

namespace solver {

Evaluator::Evaluator(const GraphInfo &graph, const SearchConfig &config,
					 const ScheduleState &state, const Candidate &candidate)
	: graph_(graph), config_(config), state_(state), candidate_(candidate) {
	summarize_candidate();
}

std::optional<CandidateEvaluation> Evaluator::evaluate_candidate() {
	const auto &tip = graph_.problem().tensors.at(summary_.tip_output_tensor);
	const auto w_choices =
		make_dim_choices(tip.width, graph_.problem().native_granularity.width);
	const auto h_choices =
		make_dim_choices(tip.height, graph_.problem().native_granularity.height);
	const auto spatial_choices = make_spatial_choices(w_choices, h_choices);

	if (summary_.single_matmul) {
		return evaluate_single_matmul_candidate(spatial_choices);
	}
	if (summary_.matmul_pointwise_chain) {
		return evaluate_matmul_pointwise_chain_candidate(spatial_choices);
	}
	if (summary_.all_pointwise) {
		return evaluate_pointwise_candidate(spatial_choices);
	}

	std::optional<CandidateEvaluation> best;
	double best_latency = std::numeric_limits<double>::infinity();
	for (const auto &[w, h] : spatial_choices) {
		for (int64_t k : make_k_choices()) {
			auto eval = evaluate_at_granularity(mlsys::Granularity{w, h, k});
			if (!eval.has_value() || eval->newly_covered_ops == 0) {
				continue;
			}
			if (eval->subgraph.subgraph_latency < best_latency) {
				best_latency = eval->subgraph.subgraph_latency;
				best = std::move(eval);
			}
		}
	}
	return best;
}

std::optional<CandidateEvaluation> Evaluator::evaluate_pointwise_candidate(
	const std::vector<std::pair<int64_t, int64_t>> &spatial_choices) {
	std::optional<CandidateEvaluation> best_for_area;
	double best_latency_for_area = std::numeric_limits<double>::infinity();
	int64_t current_area = -1;

	for (const auto &[w, h] : spatial_choices) {
		const int64_t area = w * h;
		if (current_area != -1 && area != current_area &&
			best_for_area.has_value()) {
			return best_for_area;
		}
		current_area = area;

		auto eval = evaluate_at_granularity(mlsys::Granularity{w, h, 1});
		if (!eval.has_value() || eval->newly_covered_ops == 0) {
			continue;
		}
		if (eval->subgraph.subgraph_latency < best_latency_for_area) {
			best_latency_for_area = eval->subgraph.subgraph_latency;
			best_for_area = std::move(eval);
		}
	}
	return best_for_area;
}

std::optional<CandidateEvaluation> Evaluator::evaluate_single_matmul_candidate(
	const std::vector<std::pair<int64_t, int64_t>> &spatial_choices) {
	const auto k_choices = make_k_choices();
	std::optional<CandidateEvaluation> best_for_area;
	double best_latency_for_area = std::numeric_limits<double>::infinity();
	int64_t current_area = -1;

	for (const auto &[w, h] : spatial_choices) {
		const int64_t area = w * h;
		if (current_area != -1 && area != current_area &&
			best_for_area.has_value()) {
			return best_for_area;
		}
		current_area = area;

		for (int64_t k : k_choices) {
			auto eval = evaluate_at_granularity(mlsys::Granularity{w, h, k});
			if (!eval.has_value() || eval->newly_covered_ops == 0) {
				continue;
			}
			if (eval->subgraph.subgraph_latency < best_latency_for_area) {
				best_latency_for_area = eval->subgraph.subgraph_latency;
				best_for_area = std::move(eval);
			}
			// The first feasible k is the largest k for this (w,h).
			break;
		}
	}
	return best_for_area;
}

std::optional<CandidateEvaluation>
Evaluator::evaluate_matmul_pointwise_chain_candidate(
	const std::vector<std::pair<int64_t, int64_t>> &spatial_choices) {
	return evaluate_single_matmul_candidate(spatial_choices);
}

std::optional<CandidateEvaluation>
Evaluator::evaluate_at_granularity(const mlsys::Granularity &g) {
	if (g.width <= 0 || g.height <= 0 || g.depth <= 0) {
		return std::nullopt;
	}

	const auto &tip_tensor =
		graph_.problem().tensors.at(summary_.tip_output_tensor);
	const int64_t rows = ceil_div(tip_tensor.height, g.height);
	const int64_t cols = ceil_div(tip_tensor.width, g.width);

	const auto raster = build_raster_order(rows, cols);
	const auto snake = build_snake_order(rows, cols);
	const bool use_snake_for_matmul = summary_.has_matmul;
	const auto &order = use_snake_for_matmul ? snake : raster;

	const auto retention_choices = enumerate_retention_choices(
		graph_, state_, summary_.boundary_outputs, config_.max_retained_outputs);

	std::optional<CandidateEvaluation> best;
	double best_latency = std::numeric_limits<double>::infinity();

	for (const auto &retain : retention_choices) {
		TileSimulator simulator(graph_, state_, candidate_,
								summary_.external_inputs,
								summary_.boundary_outputs,
								summary_.tip_output_tensor, g, retain.tensors);
		auto result = simulator.simulate(order);
		if (!result.has_value()) {
			continue;
		}

		std::optional<mlsys::TraversalOrder> traversal;
		if (use_snake_for_matmul) {
			traversal = snake;
		}

		if (result->latency < best_latency) {
			best_latency = result->latency;
			best = make_evaluation(g, retain.tensors, traversal, result->latency);
		}
	}
	return best;
}

CandidateEvaluation Evaluator::make_evaluation(
	const mlsys::Granularity &g, const std::vector<size_t> &retained_tensors,
	const std::optional<mlsys::TraversalOrder> &traversal,
	double latency) const {
	CandidateEvaluation eval;
	eval.boundary_outputs = summary_.boundary_outputs;
	eval.subgraph.ops = candidate_.ops;
	eval.subgraph.tensors_to_retain = retained_tensors;
	eval.subgraph.granularity = g;
	eval.subgraph.traversal_order = traversal;
	eval.subgraph.subgraph_latency = latency;
	eval.newly_covered_ops = 0;
	for (size_t op : candidate_.ops) {
		if (!state_.covered_ops.at(op)) {
			++eval.newly_covered_ops;
		}
	}
	return eval;
}

void Evaluator::summarize_candidate() {
	summary_.boundary_outputs = graph_.subset_boundary_outputs(candidate_.ops);
	summary_.external_inputs = graph_.subset_external_inputs(candidate_.ops);
	summary_.tip_output_tensor = graph_.single_output(candidate_.tip_op);

	std::unordered_set<size_t> op_set(candidate_.ops.begin(),
									  candidate_.ops.end());
	summary_.linear = true;
	for (size_t op : candidate_.ops) {
		int in_count = 0;
		int out_count = 0;
		for (size_t p : graph_.pred_ops().at(op)) {
			if (op_set.count(p)) {
				++in_count;
			}
		}
		for (size_t c : graph_.succ_ops().at(op)) {
			if (op_set.count(c)) {
				++out_count;
			}
		}
		if (in_count > 1 || out_count > 1) {
			summary_.linear = false;
			break;
		}
	}

	summary_.has_matmul = false;
	summary_.all_pointwise = true;
	for (size_t op_id : candidate_.ops) {
		if (graph_.is_matmul(op_id)) {
			summary_.has_matmul = true;
			summary_.all_pointwise = false;
		} else if (!graph_.is_pointwise(op_id)) {
			summary_.all_pointwise = false;
		}
	}
	summary_.single_matmul =
		(candidate_.ops.size() == 1 && graph_.is_matmul(candidate_.ops[0]));
	summary_.matmul_pointwise_chain =
		(candidate_.kind == CandidateType::MatMulPointwiseChain);
}

std::vector<int64_t> Evaluator::build_raster_order(int64_t rows,
												   int64_t cols) {
	std::vector<int64_t> order;
	order.reserve(static_cast<size_t>(rows * cols));
	for (int64_t r = 0; r < rows; ++r) {
		for (int64_t c = 0; c < cols; ++c) {
			order.push_back(r * cols + c);
		}
	}
	return order;
}

std::vector<int64_t> Evaluator::build_snake_order(int64_t rows,
												  int64_t cols) {
	std::vector<int64_t> order;
	order.reserve(static_cast<size_t>(rows * cols));
	for (int64_t r = 0; r < rows; ++r) {
		if ((r % 2) == 0) {
			for (int64_t c = 0; c < cols; ++c) {
				order.push_back(r * cols + c);
			}
		} else {
			for (int64_t c = cols - 1; c >= 0; --c) {
				order.push_back(r * cols + c);
			}
		}
	}
	return order;
}

int64_t Evaluator::ceil_div(int64_t a, int64_t b) {
	return (a + b - 1) / b;
}

std::vector<int64_t> Evaluator::make_dim_choices(int64_t dim,
												 int64_t native_dim) {
	std::vector<int64_t> choices;
	if (dim <= 0 || native_dim <= 0) {
		return choices;
	}

	int64_t x = std::min<int64_t>(dim, native_dim);
	while (x >= 1) {
		choices.push_back(x);
		if (x == 1) {
			break;
		}
		x = std::max<int64_t>(1, x / 2);
	}

	std::sort(choices.begin(), choices.end());
	choices.erase(std::unique(choices.begin(), choices.end()), choices.end());
	std::sort(choices.rbegin(), choices.rend());
	return choices;
}

std::vector<std::pair<int64_t, int64_t>>
Evaluator::make_spatial_choices(const std::vector<int64_t> &w_choices,
								 const std::vector<int64_t> &h_choices) const {
	std::vector<std::pair<int64_t, int64_t>> choices;
	for (int64_t w : w_choices) {
		for (int64_t h : h_choices) {
			choices.push_back({w, h});
		}
	}
	std::sort(choices.begin(), choices.end(),
			  [](const auto &a, const auto &b) {
				  const int64_t area_a = a.first * a.second;
				  const int64_t area_b = b.first * b.second;
				  if (area_a != area_b) {
					  return area_a > area_b;
				  }
				  if (a.first != b.first) {
					  return a.first > b.first;
				  }
				  return a.second > b.second;
			  });
	return choices;
}

std::vector<int64_t> Evaluator::make_k_choices() {
	if (!summary_.has_matmul) {
		return {1};
	}

	int64_t max_k =
		std::max<int64_t>(1, graph_.problem().native_granularity.width);
	for (size_t op_id : candidate_.ops) {
		if (graph_.is_matmul(op_id)) {
			max_k = std::min<int64_t>(max_k, graph_.get_inner_dim(op_id));
		}
	}

	std::vector<int64_t> choices;
	for (int64_t k = max_k; k >= 1;) {
		choices.push_back(k);
		if (k == 1) {
			break;
		}
		k = std::max<int64_t>(1, k / 2);
	}

	std::sort(choices.begin(), choices.end());
	choices.erase(std::unique(choices.begin(), choices.end()), choices.end());
	std::sort(choices.rbegin(), choices.rend());
	return choices;
}

} // namespace solver
