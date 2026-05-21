/* BeamSearchScheduler expands schedule states, evaluates candidates, applies memory-state transitions and prunes the beam. */

#include "state_search.h"
#include "cost_model.h"
#include "debug.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace solver {

	/* Convert array of bool into string, used as key later */
	static std::string bits_to_string(const std::vector<bool> &bits) {
		std::string s;
		s.reserve(bits.size());
		for (bool b : bits) {
			s.push_back(b ? '1' : '0');
		}
		return s;
	}

	/* Create an unique string for a search state */
	static std::string state_key(const ScheduleState &state) {
		return bits_to_string(state.covered_ops) + "|" +
			   bits_to_string(state.stored_tensors) + "|" +
			   bits_to_string(state.resident_tensors);
	}

	static bool consumes_required_residents(const GraphInfo &graph,
											const ScheduleState &state,
											const Candidate &candidate) {
		std::unordered_set<size_t> candidate_ops(candidate.ops.begin(),
												 candidate.ops.end());
		for (size_t t = 0; t < state.resident_tensors.size(); ++t) {
			if (!state.resident_tensors.at(t) || state.stored_tensors.at(t)) {
				continue;
			}

			for (size_t consumer : graph.consumers_of_tensor().at(t)) {
				if (state.covered_ops.at(consumer)) {
					continue;
				}
				if (!candidate_ops.count(consumer)) {
					return false;
				}
			}
		}
		return true;
	}

	/* Heuristic function, return estimated total latency */
	double BeamSearchScheduler::estimated_total_latency(
		const ScheduleState &state) const {
		double remaining = 0.0;
		for (size_t op_id = 0; op_id < graph_.problem().ops.size(); ++op_id) {
			if (!state.covered_ops.at(op_id)) {
				// Simple future cost estimation
				remaining += 0.10 * static_cast<double>(
										graph_.problem().ops[op_id].base_cost);
			}
		}
		return state.total_latency + remaining;
	}

	/* Check if all graph outputs are stored */
	bool BeamSearchScheduler::is_all_outputs_stored(
		const ScheduleState &state) const {
		for (size_t t = 0; t < graph_.problem().tensors.size(); ++t) {
			if (graph_.is_graph_output().at(t) && !state.stored_tensors.at(t)) {
				return false;
			}
		}
		return true;
	}

	mlsys::Solution BeamSearchScheduler::build_solution() const {

		/* Initial state */
		ScheduleState initial;

		initial.covered_ops.assign(graph_.problem().ops.size(), false);
		initial.stored_tensors.assign(graph_.problem().tensors.size(), false);
		initial.resident_tensors.assign(graph_.problem().tensors.size(), false);

		for (size_t t = 0; t < graph_.problem().tensors.size(); ++t) {
			if (graph_.is_graph_input().at(t)) {
				initial.stored_tensors[t] = true;
			}
		}

		/* Beam search */
		std::vector<ScheduleState> beam{initial};
		std::optional<ScheduleState> best_complete;

		DEBUG_LOG("Start search with beam width %d, graph size %zu\n",
				  config_.beam_width, graph_.problem().ops.size());
		for (int step = 0; step < config_.max_search_steps; step++) {

			DEBUG_LOG("Search step %d, beam size: %zu\n", step, beam.size());
			for (const auto &state : beam) {
				if (state.covered_count ==
						static_cast<int>(graph_.problem().ops.size()) &&
					is_all_outputs_stored(state)) {
					if (!best_complete.has_value() ||
						state.total_latency < best_complete->total_latency) {
						best_complete = state;
					}
				}
			}
			if (best_complete.has_value()) {
				return mlsys::Solution{best_complete->schedule};
			}

			std::vector<ScheduleState> expanded_beam;
			for (const auto &state : beam) {
				/* Search for schedule path instead of building subgraph
				 * directly */
				auto candidates = generator_.generate(state);
				std::sort(candidates.begin(), candidates.end(),
						  [&](const Candidate &a, const Candidate &b) {
							  // Larger candidate first
							  if (a.ops.size() != b.ops.size())
								  return a.ops.size() > b.ops.size();
							  // Without recomputation is preferred
							  if (a.uses_recompute != b.uses_recompute)
								  return !a.uses_recompute;
							  return a.tip_op < b.tip_op; // For stability
						  });
				// Limit the number of candiates
				if (candidates.size() > config_.max_candidate_num) {
					candidates.resize(config_.max_candidate_num);
				}

				/* Building subgraphs from candidates */
				for (const auto &candidate : candidates) {
					if (!consumes_required_residents(graph_, state, candidate)) {
						continue;
					}
					Evaluator evaluator(graph_, config_, state, candidate);
					auto eval = evaluator.evaluate_candidate();
					if (!eval.has_value()) {
						continue;
					}

					ScheduleState next_state = state;
					next_state.total_latency += eval->subgraph.subgraph_latency;
					next_state.schedule.push_back(eval->subgraph);

					/* Rebuild fast memory resident tensors */
					std::vector<bool> next_resident(
						graph_.problem().tensors.size(), false);
					for (size_t t : eval->subgraph.tensors_to_retain) {
						next_resident[t] = true;
					}
					next_state.resident_tensors.swap(next_resident);

					/* Previous resident tensors are only available across the
					 * current boundary. The retain list in the output format
					 * applies to outputs of this subgraph, so old residents must
					 * not be silently written back to slow memory for free. */

					for (size_t t : eval->boundary_outputs) {
						if (!next_state.resident_tensors.at(t)) {
							next_state.stored_tensors[t] = true;
						}
					}

					for (size_t op_id : candidate.ops) {
						if (!next_state.covered_ops.at(op_id)) {
							next_state.covered_ops[op_id] = true;
							next_state.covered_count++;
						}
					}
					expanded_beam.push_back(std::move(next_state));
				}
			}

			if (expanded_beam.empty()) {
				throw std::runtime_error(
					"Search got stuck when trying to expand states");
			}

			/* Retain the best schdule to the same state */
			std::unordered_map<std::string, ScheduleState> best_beam_map;
			for (auto &state : expanded_beam) {
				const std::string key = state_key(state);
				auto it = best_beam_map.find(key);
				if (it == best_beam_map.end() ||
					state.total_latency < it->second.total_latency) {
					best_beam_map[key] = std::move(state);
				}
			}

			beam.clear();
			beam.reserve(best_beam_map.size());
			for (auto &key_value_pair : best_beam_map) {
				beam.push_back(std::move(key_value_pair.second));
			}

			std::sort(beam.begin(), beam.end(),
					  [&](const ScheduleState &a, const ScheduleState &b) {
						  if (estimated_total_latency(a) !=
							  estimated_total_latency(b)) {
							  return estimated_total_latency(a) <
									 estimated_total_latency(b);
						  }
						  return a.covered_count > b.covered_count;
					  });
			if (static_cast<int>(beam.size()) > config_.beam_width) {
				beam.resize(config_.beam_width);
			}
		}

		if (best_complete.has_value()) {
			return mlsys::Solution{best_complete->schedule};
		}
		throw std::runtime_error(
			"Search exceeded max_search_steps before completion");
	}

} // namespace solver
