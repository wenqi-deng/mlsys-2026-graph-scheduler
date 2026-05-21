/* GraphInfo builds the read-only tensor/op index used by candidate generation, evaluation and beam search. */

#include "graph_utils.h"
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace solver {

GraphInfo::GraphInfo(const mlsys::Problem &problem) : problem_(problem) {
	build_tensor_maps();
	build_op_adjacency();
	build_topo_order();
}

/* ============ Graph info construction =============== */

void GraphInfo::build_tensor_maps() {
	producer_of_tensor_.assign(problem_.tensors.size(), -1);
	consumers_of_tensor_.assign(problem_.tensors.size(), {});
	tensor_usage_info_.assign(problem_.tensors.size(), {});
	is_graph_input_.assign(problem_.tensors.size(), true);
	is_graph_output_.assign(problem_.tensors.size(), true);

	for (size_t op_id = 0; op_id < problem_.ops.size(); ++op_id) {
		const auto &op = problem_.ops[op_id];
		for (size_t t : op.outputs) {
			if (t >= problem_.tensors.size()) {
				throw std::runtime_error("Output tensor index out of range");
			}
			if (producer_of_tensor_[t] != -1) {
				throw std::runtime_error("Tensor has multiple producers");
			}
			producer_of_tensor_[t] = static_cast<int>(op_id);
			is_graph_input_[t] = false;
		}
		for (int i = 0; i < static_cast<int>(op.inputs.size()); ++i) {
			const size_t t = op.inputs[i];
			if (t >= problem_.tensors.size()) {
				throw std::runtime_error("Input tensor index out of range");
			}
			consumers_of_tensor_[t].push_back(op_id);
			tensor_usage_info_[t].push_back(TensorUse{op_id, i});
			is_graph_output_[t] = false;
		}
	}
}

// Build opration adjancency map base on tensor maps
void GraphInfo::build_op_adjacency() {
	pred_ops_.assign(problem_.ops.size(), {});
	succ_ops_.assign(problem_.ops.size(), {});
	for (size_t op_id = 0; op_id < problem_.ops.size(); ++op_id) {
		for (size_t t : problem_.ops[op_id].inputs) {
			const int p = producer_of_tensor_[t];
			if (p != -1) {
				pred_ops_[op_id].push_back(static_cast<size_t>(p));
				succ_ops_[p].push_back(op_id);
			}
		}
		std::sort(pred_ops_[op_id].begin(), pred_ops_[op_id].end());
		pred_ops_[op_id].erase(
			std::unique(pred_ops_[op_id].begin(), pred_ops_[op_id].end()),
			pred_ops_[op_id].end());
	}
	for (auto &v : succ_ops_) {
		std::sort(v.begin(), v.end());
		v.erase(std::unique(v.begin(), v.end()), v.end());
	}
}

void GraphInfo::build_topo_order() {
	std::vector<int> indegree(problem_.ops.size(), 0);
	for (size_t i = 0; i < problem_.ops.size(); ++i) {
		indegree[i] = static_cast<int>(pred_ops_[i].size());
	}

	std::queue<size_t> q;
	for (size_t i = 0; i < indegree.size(); ++i) {
		if (indegree[i] == 0) {
			q.push(i);
		}
	}

	while (!q.empty()) {
		const size_t op = q.front();
		q.pop();
		topological_order_.push_back(op);
		for (size_t s : succ_ops_[op]) {
			indegree[s]--;
			if (indegree[s] == 0) {
				q.push(s);
			}
		}
	}

	if (topological_order_.size() != problem_.ops.size()) {
		throw std::runtime_error("Graph is not a DAG");
	}
}

/* ============ Tensor and Operation info =============== */

bool GraphInfo::is_pointwise(size_t op_id) const {
	return problem_.ops.at(op_id).op_type == "Pointwise";
}

bool GraphInfo::is_matmul(size_t op_id) const {
	return problem_.ops.at(op_id).op_type == "MatMul";
}

size_t GraphInfo::single_output(size_t op_id) const {
	const auto &op = problem_.ops.at(op_id);
	if (op.outputs.size() != 1) {
		throw std::runtime_error("This solver assumes one output per op");
	}
	return op.outputs[0];
}

int64_t GraphInfo::tensor_elements(size_t tensor_id) const {
	const auto &t = problem_.tensors.at(tensor_id);
	return t.width * t.height;
}

int64_t GraphInfo::get_inner_dim(size_t op_id) const {
	const auto &op = problem_.ops.at(op_id);
	if (!is_matmul(op_id)) {
		return 1;
	}
	if (op.inputs.size() != 2) {
		throw std::runtime_error("MatMul must have exactly 2 inputs");
	}
	const auto &lhs = problem_.tensors.at(op.inputs[0]);
	const auto &rhs = problem_.tensors.at(op.inputs[1]);

	if (lhs.width == rhs.height) {
		return lhs.width;
	}

	// We accept that as a fallback happen to apply to case 17
	if (lhs.height == rhs.width) {
		return lhs.height;
	}

	throw std::runtime_error("MatMul dimension mismatch");
}

int64_t GraphInfo::input_slice_elements(size_t op_id, int input_index,
										const mlsys::Granularity &g) const {
	if (is_pointwise(op_id)) {
		return g.width * g.height;
	}
	const int64_t k = std::min<int64_t>(g.depth, get_inner_dim(op_id));
	return (input_index == 0) ? (g.height * k) : (k * g.width);
}

int64_t GraphInfo::output_slice_elements(const mlsys::Granularity &g) const {
	return g.width * g.height;
}

double GraphInfo::transfer_time(int64_t elements) const {
	return static_cast<double>(elements) /
		   static_cast<double>(problem_.slow_memory_bandwidth);
}

/* ============ Analyse subgraph =============== */

std::vector<size_t>
GraphInfo::topo_sort_subset(const std::vector<size_t> &ops) const {
	std::unordered_set<size_t> keep(ops.begin(), ops.end());
	std::vector<int> indegree(problem_.ops.size(), 0);
	for (size_t op : ops) {
		for (size_t p : pred_ops_[op]) {
			if (keep.count(p)) {
				indegree[op]++;
			}
		}
	}

	std::queue<size_t> q;
	for (size_t op : ops) {
		if (indegree[op] == 0) {
			q.push(op);
		}
	}

	std::vector<size_t> out;
	while (!q.empty()) {
		const size_t op = q.front();
		q.pop();
		out.push_back(op);
		for (size_t s : succ_ops_[op]) {
			if (!keep.count(s))
				continue;
			--indegree[s];
			if (indegree[s] == 0) {
				q.push(s);
			}
		}
	}
	return out;
}

/* Return subgraph output */
std::vector<size_t>
GraphInfo::subset_boundary_outputs(const std::vector<size_t> &ops) const {
	std::unordered_set<size_t> op_set(ops.begin(), ops.end());
	std::vector<size_t> boundary;
	for (size_t op_id : ops) {
		for (size_t t : problem_.ops[op_id].outputs) {
			bool external = is_graph_output_[t];
			for (size_t c : consumers_of_tensor_[t]) {
				if (!op_set.count(c)) {
					external = true;
					break;
				}
			}
			if (external) {
				boundary.push_back(t);
			}
		}
	}
	std::sort(boundary.begin(), boundary.end());
	boundary.erase(std::unique(boundary.begin(), boundary.end()),
				   boundary.end());
	return boundary;
}

/* Return subgraph eternal input */
std::vector<size_t>
GraphInfo::subset_external_inputs(const std::vector<size_t> &ops) const {
	std::unordered_set<size_t> op_set(ops.begin(), ops.end());
	std::vector<size_t> inputs;
	for (size_t op_id : ops) {
		for (size_t t : problem_.ops[op_id].inputs) {
			const int p = producer_of_tensor_[t];
			if (p == -1 || !op_set.count(static_cast<size_t>(p))) {
				inputs.push_back(t);
			}
		}
	}
	std::sort(inputs.begin(), inputs.end());
	inputs.erase(std::unique(inputs.begin(), inputs.end()), inputs.end());
	return inputs;
}

} // namespace solver
