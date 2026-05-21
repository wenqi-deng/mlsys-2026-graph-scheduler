#ifndef PROBLEM_IO_H_
#define PROBLEM_IO_H_

#include "json.hpp"
#include "mlsys.h"
#include <cassert>

using json = nlohmann::json;

namespace solver {

mlsys::Problem read_problem(const std::string &filename);

void write_solution(const std::string &filename,
					const mlsys::Solution &solution);

} // namespace solver

#endif
