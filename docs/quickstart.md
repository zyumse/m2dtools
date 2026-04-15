Load and inspect a LAMMPS full-style data file:

```python
from m2dtools.lmp import read_lammps_full

# m2dtools uses a Lammps class to store the data
lmp = read_lammps_full("system.data")

print(lmp.natoms)  # number of atoms
print(lmp.atom_info) # lammps atom info part
if lmp.nbonds > 0:
    print(lmp.bond_info) # lammps bond info part
```

Compute an RDF from coordinates:

```python
import m2dtools.basic.
rdf, r = compute_rdf(positions, box, dr=0.01, rmax=20.0)
```
