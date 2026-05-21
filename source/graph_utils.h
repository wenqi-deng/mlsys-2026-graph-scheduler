#ifndef GRAPH_UTILS_H_
#define GRAPH_UTILS_H_

#include "mlsys.h"
#include <cstdint>
#include <vector>

namespace solver {

// Represents how a tensor is used in an operation
struct TensorUse {
	size_t op_id;	 //  Operation ID
	int input_index; //  Input index in that operation
};

class GraphInfo {
  public:
	explicit GraphInfo(const mlsys::Problem &problem);

	const mlsys::Problem &problem() const {
		return problem_;
	}
	const std::vector<int> &producer_of_tensor() const {
		return producer_of_tensor_;
	}
	const std::vector<std::vector<size_t>> &consumers_of_tensor() const {
		return consumers_of_tensor_;
	}
	const std::vector<std::vector<TensorUse>> &tensor_uses() const {
		return tensor_usage_info_;
	}
	const std::vector<std::vector<size_t>> &pred_ops() const {
		return pred_ops_;
	}
	const std::vector<std::vector<size_t>> &succ_ops() const {
		return succ_ops_;
	}
	const std::vector<size_t> &topological_order() const {
		return topological_order_;
	}
	const std::vector<bool> &is_graph_input() const {
		return is_graph_input_;
	}
	const std::vector<bool> &is_graph_output() const {
		return is_graph_output_;
	}

	bool is_pointwise(size_t op_id) const;
	bool is_matmul(size_t op_id) const;
	size_t single_output(size_t op_id) const;
	int64_t tensor_elements(size_t tensor_id) const;
	int64_t get_inner_dim(size_t op_id) const;
	int64_t input_slice_elements(size_t op_id, int input_index,
								 const mlsys::Granularity &g) const;
	int64_t output_slice_elements(const mlsys::Granularity &g) const;
	double transfer_time(int64_t elements) const;

	std::vector<size_t> topo_sort_subset(const std::vector<size_t> &ops) const;
	std::vector<size_t>
	subset_boundary_outputs(const std::vector<size_t> &ops) const;
	std::vector<size_t>
	subset_external_inputs(const std::vector<size_t> &ops) const;

  private:
	void build_tensor_maps();
	void build_op_adjacency();
	void build_topo_order();

	mlsys::Problem problem_;
	std::vector<int> producer_of_tensor_;
	std::vector<std::vector<size_t>> consumers_of_tensor_;
	std::vector<std::vector<TensorUse>> tensor_usage_info_;
	std::vector<std::vector<size_t>> pred_ops_;
	std::vector<std::vector<size_t>> succ_ops_;
	std::vector<size_t> topological_order_;
	std::vector<bool> is_graph_input_;
	std::vector<bool> is_graph_output_;
};

} // namespace solver

#endif
