import numpy as np
import m2dtools.lmp.tools_lammps as tl_lmp
import matplotlib.pyplot as plt

class DielectricWithP:
    def __init__(self, lmp_file):
        self.lmp = tl_lmp.read_lammps_full(lmp_file)
        self.lmp.atom_info = self.lmp.atom_info[np.argsort(self.lmp.atom_info[:,0])]
        self.layer_boundaries = self.get_layer_boundary()
        self.volume_per_layer = self.compute_volume_per_layer()
        self.z_layers_center, self.layer_polarizations = self.compute_layer_polarizations()

    def get_layer_boundary(self):
        atom_types = self.lmp.atom_info[:,2]
        ti_indices = np.where(atom_types == 1)[0]
        ti_z_positions = self.lmp.atom_info[ti_indices, 6]

        # find Ti layer positions
        ti_z_sorted = np.sort(ti_z_positions)
        layers = []
        current_layer = [ti_z_sorted[0]]
        for z in ti_z_sorted[1:]:
            if abs(z - current_layer[-1]) <= 0.5:
                current_layer.append(z)
            else:
                layers.append(np.mean(current_layer))
                current_layer = [z]
        layers.append(np.mean(current_layer))
        ti_layer_positions = np.array(layers)
        print(f'num of Ti layers: {len(ti_layer_positions)}')

        boundaries = []
        for i in range(len(ti_layer_positions) - 1):
            d = ti_layer_positions[i+1] - ti_layer_positions[i]
            if d > 1.5:
                boundary = (ti_layer_positions[i+1] + ti_layer_positions[i]) / 2
                boundaries.append(boundary)
        boundaries = np.array(boundaries)
        print('num of boundaries:', len(boundaries))
        self.num_layers = int(len(ti_layer_positions)/2)
        return boundaries

    def compute_volume_per_layer(self):
        lmp = self.lmp
        num_layers = self.num_layers
        # Volume of tio2 slab
        area_xy = (lmp.x[1]-lmp.x[0]) * (lmp.y[1]-lmp.y[0])
        z_positions = lmp.atom_info[:, 6]
        Lz = lmp.z[1] - lmp.z[0]
        z_half_Lz = Lz / 2
        z_surf1 = np.max(z_positions[z_positions < z_half_Lz])
        z_surf2 = np.min(z_positions[z_positions > z_half_Lz])
        volume = area_xy * (Lz-z_surf2 + z_surf1)
        print(f'Slab surfaces: {z_surf1} A, {z_surf2} A')
        print(f'TiO2 volume: {volume} A^3')
        # intotal 15 layers
        print(f'Number of layers: {num_layers}')
        volume_per_layer = volume / num_layers
        print(f'Volume per layer: {volume_per_layer} A^3')
        return volume_per_layer

    def compute_layer_polarizations(self):
        lmp = self.lmp
        layer_boundaries = self.layer_boundaries
        volume_per_layer = self.volume_per_layer
        atom_types = lmp.atom_info[:,2]
        ti_indices = np.where(atom_types == 1)[0]
        ti_z_positions = lmp.atom_info[ti_indices, 6]

        # if layer_boundaries
        if layer_boundaries[-1] - lmp.z[1] < -3:
            layer_boundaries = np.concatenate((layer_boundaries, lmp.z[1:]))

        # find Ti layer positions
        ti_z_sorted = np.sort(ti_z_positions)
        layers = []
        current_layer = [ti_z_sorted[0]]
        for z in ti_z_sorted[1:]:
            if abs(z - current_layer[-1]) <= 0.5:
                current_layer.append(z)
            else:
                layers.append(np.mean(current_layer))
                current_layer = [z]
        layers.append(np.mean(current_layer))
        ti_layer_positions = np.array(layers)

        num_layers = len(layer_boundaries)
        print('Number of layers:', num_layers)
        atom_layers = -1 * np.ones(len(lmp.atom_info), dtype=int)
        for i in range(len(lmp.atom_info)):
            z = lmp.atom_info[i, 6]
            for j in range(num_layers):
                if z < layer_boundaries[j]:
                    atom_layers[i] = j
                    break
            if atom_layers[i] == -1:
                atom_layers[i] = 0
        self.atom_layers = atom_layers

        z_layers_center = np.zeros(num_layers)
        for j in range(num_layers - 1):
            z_positions_Ti_in_layer = ti_layer_positions[(ti_layer_positions > layer_boundaries[j-1]) & (ti_layer_positions < layer_boundaries[j])] if j > 0 else ti_layer_positions[ti_layer_positions < layer_boundaries[j]]
            z_layers_center[j] = np.mean(z_positions_Ti_in_layer)
        Lz = lmp.z[1] - lmp.z[0]
        z_positions_Ti_in_last_layer = ti_layer_positions[ti_layer_positions > layer_boundaries[-1]]
        z_layers_center[-1] = np.mean(np.concatenate((z_positions_Ti_in_last_layer, ti_layer_positions[ti_layer_positions < layer_boundaries[0]] + Lz)))

        # make sure charge neutrality in each layer
        for j in range(num_layers):
            layer_charge = np.sum(lmp.atom_info[atom_layers == j, 3])
            # warning if not neutral
            if abs(layer_charge) > 1e-5:
                print(f'Warning: Layer {j} has non-zero charge: {layer_charge} e')
        

        layer_polarizations = np.zeros((num_layers, 3))
        # align atom within a layer to a reference
        for j in range(num_layers):
            indices = np.where(atom_layers == j)[0]
            if len(indices) == 0:
                continue
            z_ref = lmp.atom_info[indices[0], 6]
            for i in indices:
                z = lmp.atom_info[i, 6]
                dz = z - z_ref
                Lz = lmp.z[1] - lmp.z[0]
                if dz > Lz / 2:
                    z -= Lz
                elif dz < -Lz / 2:
                    z += Lz
                lmp.atom_info[i, 6] = z
                charge = lmp.atom_info[i, 3]
                position = lmp.atom_info[i, 4:7]
                layer_polarizations[j] += charge * position # this is M, e * Ang
        
        epsilon0 = 55.26349406 # e/V/micrometer
        layer_polarizations = layer_polarizations/volume_per_layer/epsilon0*10000 # convert to V/A

        return z_layers_center, layer_polarizations