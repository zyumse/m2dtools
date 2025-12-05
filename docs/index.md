# Welcome to m2dtools Documentation

*m2dtools* is a lightweight Python toolkit for molecular simulation workflows, providing
utilities for structure handling, trajectory analysis, coarse-graining, and LAMMPS
interfacing. It is designed to support multiscale molecular modeling with simple,
modular components that can be used independently or combined into automated
pipelines.

## Features

- **Structure and trajectory utilities**  
  Read, write, filter, and manipulate molecular data with minimal overhead.

- **Coarse-graining support**  
  Tools for computing RDFs, bonded distributions, and preparing inputs for CG
  force-field optimization workflows.

- **LAMMPS helpers**  
  Read full-style data files, extract atom and topology information, and prepare
  simulation inputs.

- **Network and graph tools**  
  Construct connectivity networks, compute graph-based descriptors, and analyze
  polymer or molecular connectivity.

- **Modular design**  
  Each submodule is self-contained, making it easy to integrate with existing
  workflows in MD, CGMD, or ML-accelerated simulations.

## Installation

```bash
pip install m2dtools
```

Or install from source:
```
git clone https://github.com/zyumse/m2dtools
cd m2dtools
pip install -e .
```
