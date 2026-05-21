#include "problem_io.h"
#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace solver {
mlsys::Problem read_problem(const std::string &filename) {

	json json;
	mlsys::Problem prob;
	std::ifstream f(filename);

	if (!f) {
		throw std::runtime_error("Cannot open input file: " + filename);
	}

	f >> json;
	auto widths = json["widths"].get<std::vector<int64_t>>();
	auto heights = json["heights"].get<std::vector<int64_t>>();

	assert(widths.size() == heights.size());

	int n = widths.size();
	prob.tensors.resize(n);

	for (int i = 0; i < n; i++) {
		prob.tensors[i].width = widths[i];
		prob.tensors[i].height = heights[i];
	}

	int m = json["inputs"].size();
	prob.ops.resize(m);

	for (int i = 0; i < m; i++) {
		mlsys::Op op;
		op.inputs = json["inputs"][i].get<std::vector<size_t>>();
		op.outputs = json["outputs"][i].get<std::vector<size_t>>();
		op.base_cost = json["base_costs"][i];
		op.op_type = json["op_types"][i];
		prob.ops[i] = op;
	}

	prob.fast_memory_capacity = json["fast_memory_capacity"];
	prob.slow_memory_bandwidth = json["slow_memory_bandwidth"];
	prob.native_granularity.width = json["native_granularity"][0];
	prob.native_granularity.height = json["native_granularity"][1];
	prob.native_granularity.depth = 1;

	return prob;
}

void write_solution(const std::string &filename,
					const mlsys::Solution &solution) {
	json j;

	for (const auto &subgraph : solution.subgraphs) {
		j["subgraphs"].push_back(subgraph.ops);

		j["granularities"].push_back({subgraph.granularity.width,
									  subgraph.granularity.height,
									  subgraph.granularity.depth});

		j["tensors_to_retain"].push_back(subgraph.tensors_to_retain);

		j["traversal_orders"].push_back(subgraph.traversal_order);

		j["subgraph_latencies"].push_back(subgraph.subgraph_latency);
	}

	std::ofstream f(filename);
	f << j.dump(1);
}

} // namespace solver
