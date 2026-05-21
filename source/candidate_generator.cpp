/* CandidateGenerator proposes structural subgraphs: single ops, pointwise chains, backward cones and recomputation cones. */

/* Propose candidates worth trying base purely on the graph structure */

#include "candidate_generator.h"
#include <sstream>
#include <string>
#include <unordered_set>

namespace solver {

static std::string candidate_key(const Candidate &c);

/* Generate all possible candidates */
std::vector<Candidate>
CandidateGenerator::generate(const ScheduleState &state) const {
	std::vector<Candidate> all;
	for (const Candidate &c : generate_linear_candidates(state)) {
		all.push_back(c);
	}
	for (const Candidate &c : generate_backward_cone_candidates(state, false)) {
		all.push_back(c);
	}
	for (const Candidate &c : generate_backward_cone_candidates(state, true)) {
		all.push_back(c);
	}

	std::vector<Candidate> dedupliated_candidates;
	std::unordered_set<std::string> seen;
	for (auto &candidate : all) {
		const std::string key = candidate_key(candidate);
		if (seen.insert(key).second) {
			dedupliated_candidates.push_back(candidate);
		}
	}
	return dedupliated_candidates;
}

/* If a tensor is stored in fast/ slow memory in the current state */
bool CandidateGenerator::tensor_available(const ScheduleState &state,
										  size_t tensor_id) const {
	return state.stored_tensors.at(tensor_id) ||
		   state.resident_tensors.at(tensor_id);
}

bool CandidateGenerator::is_external_tensors_available(
	const ScheduleState &state, const std::vector<size_t> &ops) const {
	const auto ext_inputs = graph_.subset_external_inputs(ops);
	for (size_t t : ext_inputs) {
		if (!tensor_available(state, t)) {
			if (graph_.is_graph_input().at(t)) {
				throw std::runtime_error("Graph input tensor is not stored in memory");
			}
			return false;
		}
	}
	return true;
}

bool CandidateGenerator::is_linear_extendable(const ScheduleState &state,
											  const Candidate &cur,
											  size_t next_op) const {
	// next_op is already covered
	if (state.covered_ops.at(next_op)) {
		return false;
	}

	//next_op is matmul
	if (graph_.is_matmul(next_op)) {
		return false;
	}

	/* tail tensor only flows to next_op*/
	const size_t tail = cur.ops.back();
	if (graph_.succ_ops().at(tail).size() != 1 ||
		graph_.succ_ops().at(tail)[0] != next_op) {
		return false;
	}

	/* next_op has only one predecessor */
	if (graph_.pred_ops().at(next_op).size() != 1) {
		return false;
	}

	/* Ensure tensor used in next_op is available: for future advancement */
	std::unordered_set<size_t> in_chain(cur.ops.begin(), cur.ops.end());
	for (size_t tensor : graph_.problem().ops[next_op].inputs) {
		const int p = graph_.producer_of_tensor().at(tensor);
		if (p != -1 && in_chain.count(static_cast<size_t>(p))) {
			continue;
		}
		if (!graph_.is_graph_input().at(tensor) &&
			!tensor_available(state, tensor)) {
			return false;
		}
	}
	return true;
}

/* Try to form a backward cone, if successful, in_candidate stores all ops in
 * the cone. All candidates are pointwise ops
 */
bool CandidateGenerator::expand_backward_cone(const ScheduleState &state,
											  size_t op_id, int *budget,
											  int depth, bool allow_recompute,
											  std::vector<bool> *in_candidate,
											  bool *used_recompute) const {
	if (in_candidate->at(op_id)) {
		return true;
	}

	if (graph_.is_matmul(op_id)) {
		return false;
	}
	if (*budget <= 0) {
		return false;
	}
	
	/* Budget can control the cone size */
	in_candidate->at(op_id) = true;
	(*budget)--;

	const auto &op = graph_.problem().ops.at(op_id);
	for (size_t t : op.inputs) {
		const int p = graph_.producer_of_tensor().at(t);
		if (p == -1) {
			continue;
		}

		const bool producer_covered =
			state.covered_ops.at(static_cast<size_t>(p));
		if (!producer_covered) {
			if (!expand_backward_cone(state, static_cast<size_t>(p), budget,
									  depth + 1, allow_recompute, in_candidate,
									  used_recompute)) {
				return false;
			}
			continue;
		}

		if (allow_recompute && depth < config_.max_recompute_depth) {
			// Allow recompute only once in the cone, to avoid search explosion
			*used_recompute = true;
			if (!expand_backward_cone(state, static_cast<size_t>(p), budget,
									  depth + 1, false, in_candidate,
									  used_recompute)) {
				return false;
			}
			continue;
		}

		if (!tensor_available(state, t) && !graph_.is_graph_input().at(t)) {
			return false;
		}
	}
	return true;
}

std::vector<Candidate> CandidateGenerator::generate_linear_candidates(
	const ScheduleState &state) const {
	std::vector<Candidate> out;
	for (size_t op_id : graph_.topological_order()) {
		if (state.covered_ops.at(op_id)) {
			continue;
		}

		/* To see if the operation is ready to be scheduled */
		bool ready = true;
		for (size_t t : graph_.problem().ops[op_id].inputs) {
			if (!graph_.is_graph_input().at(t) && !tensor_available(state, t)) {
				ready = false;
				break;
			}
		}
		if (!ready) {
			continue;
		}

		/* Single operation candidate */
		Candidate cur;
		cur.kind = CandidateType::SingleOp;
		cur.tip_op = op_id;
		cur.ops = {op_id};
		out.push_back(cur);

		/* Matmul won't be extended*/
		if (graph_.is_matmul(op_id)) {
			// Candidate matmul_chain = cur;
			// while (static_cast<int>(matmul_chain.ops.size()) <
			// 	   config_.max_chain_len) {
			// 	const size_t tail = matmul_chain.ops.back();
			// 	if (graph_.succ_ops().at(tail).size() != 1) {
			// 		break;
			// 	}
			// 	const size_t next_op = graph_.succ_ops().at(tail)[0];
			// 	if (!graph_.is_pointwise(next_op) ||
			// 		!is_linear_extendable(state, matmul_chain, next_op)) {
			// 		break;
			// 	}
			// 	matmul_chain.ops.push_back(next_op);
			// 	matmul_chain.tip_op = next_op;
			// 	matmul_chain.kind = CandidateType::MatMulPointwiseChain;
			// 	out.push_back(matmul_chain);
			// }
			continue;
		}
		/* Linear chain candidate */
		while (static_cast<int>(cur.ops.size()) < config_.max_chain_len) {
			const size_t tail = cur.ops.back();
			if (graph_.succ_ops().at(tail).size() != 1) {
				break;
			}
			const size_t next_op = graph_.succ_ops().at(tail)[0];
			if (!is_linear_extendable(state, cur, next_op)) {
				break;
			}
			cur.ops.push_back(next_op);
			cur.tip_op = next_op;
			cur.kind = CandidateType::LinearChain;
			out.push_back(cur);
		}
	}
	return out;
}

/* A cone structure is very likely to be benefit from fusion or recomputation,
 * we try to form small cones*/
std::vector<Candidate> CandidateGenerator::generate_backward_cone_candidates(
	const ScheduleState &state, bool allow_recompute) const {
	std::vector<Candidate> out;
	for (size_t possible_tip_op : graph_.topological_order()) {
		// tip should be uncovered pointwise op 
		if (state.covered_ops.at(possible_tip_op) or graph_.is_matmul(possible_tip_op)) {
			continue;
		}

		std::vector<bool> in_candidate(graph_.problem().ops.size(), false);
		int budget = config_.max_candidate_ops;
		bool used_recompute = false;
		if (!expand_backward_cone(state, possible_tip_op, &budget, 0,
								  allow_recompute, &in_candidate,
								  &used_recompute)) {
			continue;
		}

		std::vector<size_t> ops;
		for (size_t op_id = 0; op_id < in_candidate.size(); ++op_id) {
			if (in_candidate[op_id]) {
				ops.push_back(op_id);
			}
		}
		ops = graph_.topo_sort_subset(ops);

		if (ops.empty() || ops.size() == 1) {
			continue;
		}
		if (!is_external_tensors_available(state, ops)) {
			continue;
		}

		Candidate c;
		c.kind = used_recompute ? CandidateType::RecomputeCone
								: CandidateType::BackwardCone;
		c.tip_op = possible_tip_op;
		c.ops = ops;
		c.uses_recompute = used_recompute;
		out.push_back(c);
	}
	return out;
}

static std::string candidate_key(const Candidate &c) {
	std::ostringstream oss;
	oss << static_cast<int>(c.kind) << ':';
	for (size_t i = 0; i < c.ops.size(); i++) {
		if (i)
			oss << ',';
		oss << c.ops[i];
	}
	oss << ':' << (c.uses_recompute ? 1 : 0);
	return oss.str();
}

} // namespace solver
