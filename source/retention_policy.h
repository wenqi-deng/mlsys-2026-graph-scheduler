#ifndef RETENTION_POLICY_H_
#define RETENTION_POLICY_H_

#include "graph_utils.h"
#include "search_types.h"
#include <vector>

namespace solver {

	struct RetentionChoice {
		std::vector<size_t> tensors;
	};

	std::vector<RetentionChoice> enumerate_retention_choices(
		const GraphInfo &graph, const ScheduleState &state,
		const std::vector<size_t> &boundary_outputs, int max_retained_outputs);

} // namespace solver

#endif
