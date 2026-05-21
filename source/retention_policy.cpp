/* Retention policy ranks boundary outputs by future reuse and enumerates bounded retain-set candidates. */

#include "retention_policy.h"
#include <algorithm>
#include <utility>
#include <vector>

namespace solver {
namespace {

bool consumer_ready_with_fresh_tensor(const GraphInfo &graph,
									  const ScheduleState &state,
									  size_t fresh_tensor,
									  size_t consumer) {
	for (size_t input : graph.problem().ops.at(consumer).inputs) {
		if (input == fresh_tensor) {
			continue;
		}
		if (graph.is_graph_input().at(input) ||
			state.stored_tensors.at(input) ||
			state.resident_tensors.at(input)) {
			continue;
		}
		return false;
	}
	return true;
}

double retention_score(const GraphInfo &graph, const ScheduleState &state,
					   size_t tensor_id) {
	if (graph.is_graph_output().at(tensor_id)) {
		return -1.0;
	}

	int remaining_consumers = 0;
	int ready_consumers = 0;
	int nearest_consumer_distance = 1 << 20;
	const int producer = graph.producer_of_tensor().at(tensor_id);
	for (size_t c : graph.consumers_of_tensor().at(tensor_id)) {
		if (state.covered_ops.at(c)) {
			continue;
		}
		++remaining_consumers;
		if (!consumer_ready_with_fresh_tensor(graph, state, tensor_id, c)) {
			continue;
		}
		++ready_consumers;
		if (producer != -1) {
			nearest_consumer_distance = std::min(
				nearest_consumer_distance,
				std::max(1, static_cast<int>(c) - producer));
		}
	}

	/* A retained output is not also stored to slow memory. To avoid schedules
	 * that need the same tensor after the next boundary, retain only tensors
	 * with exactly one remaining consumer and make sure that consumer can be
	 * scheduled immediately from the current state plus this fresh tensor. */
	if (remaining_consumers != 1 || ready_consumers != 1) {
		return -1.0;
	}

	const double transfer = graph.transfer_time(graph.tensor_elements(tensor_id));
	const double immediacy =
		1.0 / static_cast<double>(std::max(1, nearest_consumer_distance));
	return transfer * 2.0 * immediacy;
}

int64_t total_elements(const GraphInfo &graph, const std::vector<size_t> &ts) {
	int64_t total = 0;
	for (size_t t : ts) {
		total += graph.tensor_elements(t);
	}
	return total;
}

} // namespace

std::vector<RetentionChoice> enumerate_retention_choices(
	const GraphInfo &graph, const ScheduleState &state,
	const std::vector<size_t> &boundary_outputs, int max_retained_outputs) {
	std::vector<RetentionChoice> out;
	out.push_back(RetentionChoice{});
	if (max_retained_outputs <= 0) {
		return out;
	}

	std::vector<std::pair<double, size_t>> ranked;
	for (size_t t : boundary_outputs) {
		if (graph.tensor_elements(t) > graph.problem().fast_memory_capacity) {
			continue;
		}
		const double score = retention_score(graph, state, t);
		if (score > 0.0) {
			ranked.push_back({score, t});
		}
	}
	std::sort(ranked.begin(), ranked.end(),
			  [](const auto &a, const auto &b) {
				  if (a.first != b.first) {
					  return a.first > b.first;
				  }
				  return a.second < b.second;
			  });

	const int limit = std::min<int>(static_cast<int>(ranked.size()), 6);
	std::vector<size_t> picks;
	for (int i = 0; i < limit; ++i) {
		picks.push_back(ranked[i].second);
	}

	for (size_t t : picks) {
		out.push_back(RetentionChoice{{t}});
	}

	if (max_retained_outputs >= 2) {
		for (int i = 0; i < static_cast<int>(picks.size()); ++i) {
			for (int j = i + 1; j < static_cast<int>(picks.size()); ++j) {
				std::vector<size_t> choice{picks[i], picks[j]};
				if (total_elements(graph, choice) <=
					graph.problem().fast_memory_capacity) {
					out.push_back(RetentionChoice{choice});
				}
			}
		}
	}
	return out;
}

} // namespace solver
