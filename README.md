# MLSys 2026 Solution — First Place 🥇

**Track A: Systems Engineering**

This repository contains our solution for the [**Google MLSys Competition 2026**](https://github.com/yarongmu-google/MLSys), awarded **First Place 🥇** in Track A (Systems Engineering) with a score of **19.22**.

**Author:** [Wenqi Deng](https://wenqi-deng.github.io/) from Peking University

## Documentation

- `source/`: Complete C++ source code.

- `report/`: Technical report and presentation materials.

- `benchmarks/`: Public benchmark data.

## Build

```bash
mkdir build && make all
```

## Usage

```bash
./build/mlsys <input.json> <output.json>
```

Example:
```bash
./build/mlsys benchmarks/example_problem.json solution.json
```

## Benchmarks

All benchmarks are available in the [official MLSys 2026 repository](https://github.com/yarongmu-google/MLSys/tree/main/benchmarks).

The solution was evaluated on hidden test sets and achieved the highest score across all Track A submissions.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

## Acknowledgement

This project uses JSON for Modern C++ (nlohmann/json), version 3.12.0.

Copyright (c) 2013-2026 Niels Lohmann
Licensed under the MIT License.