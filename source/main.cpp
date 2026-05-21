#include "problem_io.h"
#include "solver.h"
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
	if (argc != 3) {
		std::cerr << "Usage: mlsys <input.json> <output.json>\n";
		return 1;
	}

	try {
		const std::string input_path = argv[1];
		const std::string output_path = argv[2];

		const mlsys::Problem problem = solver::read_problem(input_path);
		solver::Solver solver;
		const mlsys::Solution solution = solver.solve(problem);
		solver::write_solution(output_path, solution);

		// compute total latency of the solution:
		double total_latency = 0.0;
		for (const auto &subgraph : solution.subgraphs) {
			total_latency += subgraph.subgraph_latency;
		}
		std::cout << "Total latency: " << total_latency << "\n";

		std::cout << "Solution written to " << output_path << "\n";
		return 0;
	} catch (const std::exception &ex) {
		std::cerr << "Error: " << ex.what() << "\n";
		return 2;
	}
}
