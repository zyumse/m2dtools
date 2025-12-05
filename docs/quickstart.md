Load and inspect a LAMMPS full-style data file:

```python
from m2dtools.lmp import read_lammps_full

lmp = read_lammps_full("system.data")
print(lmp.atom_info.shape)
print(lmp.bonds[:10])
```

Compute an RDF from coordinates:

```python
from m2dtools.cg import compute_rdf
rdf, r = compute_rdf(positions, box, dr=0.01, rmax=20.0)
```
