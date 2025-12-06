# m2dtools

*m2dtools* is a lightweight Python toolkit for molecular simulation workflows, including analysis helpers, coarse-graining pipelines, and LAMMPS structure preparation scripts.

📘 **Full Documentation:**  
https://zyumse.github.io/m2dtools/

## Installation

1. Clone this repository and move into it.
2. (Optional) Create and activate a Python 3.8+ environment.
3. Install the package in editable mode:

   ```bash
   pip install -e .
   ```

This pulls in the core dependencies (`numpy`, `matplotlib`, `scipy`, `pandas`). Editable mode lets you modify the source and use the changes immediately without reinstalling.

## Features

- `m2dtools.basic` – vectorised analysis functions for bonds, angles, PDFs/SQs, coordination numbers, autocorrelation, MSD, and compressibility.
- `m2dtools.lmp` – utilities to read, modify, and write LAMMPS full-data and dump files for simulation setup.
- `m2dtools.cg` – coarse-graining tools, including iterative Boltzmann inversion (IBI) scripts and polymer-specific helpers.
- `m2dtools.network` – routines for stochastic network generation and conversion to LAMMPS structures.
- `m2dtools.other` – format converters and domain-specific helpers (e.g. GROMACS parsing, local stress extraction, silica and water coarse-graining utilities).

## Contributing

Contributions are welcome.  
Please open an issue or submit a pull request on GitHub.

## License

Released under the MIT License.
