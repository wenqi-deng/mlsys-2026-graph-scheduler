#ifndef SOLVER_H_
#define SOLVER_H_

#include "search_types.h"

namespace solver {

class Solver {
  public:
	explicit Solver(SearchConfig config = SearchConfig()) : config_(config) {
	}

	mlsys::Solution solve(const mlsys::Problem &problem) const;

  private:
	SearchConfig config_;
};

} // namespace solver

#endif