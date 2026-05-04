"""
Update coordinates in a LAMMPS data file using a specific frame from a dump file.
"""

import argparse
import sys


def read_dump_frame(dump_file, frame_index):
    """Read atom coords (and optionally box) from a specific frame (0-indexed)."""
    frames = []
    with open(dump_file) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].strip() == "ITEM: TIMESTEP":
            frame = {}
            frame["timestep"] = int(lines[i + 1].strip())
            i += 2

            # number of atoms
            assert lines[i].strip() == "ITEM: NUMBER OF ATOMS"
            n_atoms = int(lines[i + 1].strip())
            frame["n_atoms"] = n_atoms
            i += 2

            # box bounds
            assert lines[i].strip().startswith("ITEM: BOX BOUNDS")
            box_line = lines[i].strip()
            frame["box_line"] = box_line
            box = []
            for _ in range(3):
                i += 1
                box.append(lines[i].strip())
            frame["box"] = box
            i += 2  # skip "ITEM: ATOMS ..."

            # parse column headers
            header = lines[i - 1].strip()  # already advanced past it
            cols = header.replace("ITEM: ATOMS", "").split()
            frame["cols"] = cols

            # atom data
            atoms = {}
            for _ in range(n_atoms):
                parts = lines[i].split()
                atom_id = int(parts[cols.index("id")])
                atoms[atom_id] = parts
                i += 1
            frame["atoms"] = atoms
            frames.append(frame)
        else:
            i += 1

    if not frames:
        sys.exit("Error: no frames found in dump file.")

    n = len(frames)
    if frame_index < 0:
        frame_index = n + frame_index
    if frame_index < 0 or frame_index >= n:
        sys.exit(f"Error: frame index {frame_index} out of range (0–{n-1}).")

    print(f"Dump file has {n} frame(s). Using frame {frame_index} "
          f"(timestep {frames[frame_index]['timestep']}).")
    return frames[frame_index]


def wrap_pbc(x, y, z, box_lo, box_hi):
    """Wrap Cartesian coordinates into the primary periodic box."""
    coords = [x, y, z]
    for i in range(3):
        L = box_hi[i] - box_lo[i]
        coords[i] = box_lo[i] + (coords[i] - box_lo[i]) % L
    return tuple(coords)


def update_data_file(data_file, frame, output_file, update_box, wrap):
    """Rewrite the data file with coordinates (and optionally box) from frame."""
    with open(data_file) as f:
        lines = f.readlines()

    cols = frame["cols"]
    if "id" not in cols:
        sys.exit("Error: dump file must contain 'id' column.")

    has_xyz = all(c in cols for c in ("x", "y", "z"))
    has_sxyz = all(c in cols for c in ("xs", "ys", "zs"))
    if not has_xyz and not has_sxyz:
        sys.exit("Error: dump file must contain x/y/z or xs/ys/zs columns.")

    # Box bounds always come from the dump frame
    raw = frame["box"]
    box_lo = [float(b.split()[0]) for b in raw]
    box_hi = [float(b.split()[1]) for b in raw]

    def get_coords(parts):
        if has_xyz:
            x = float(parts[cols.index("x")])
            y = float(parts[cols.index("y")])
            z = float(parts[cols.index("z")])
        else:
            xs = float(parts[cols.index("xs")])
            ys = float(parts[cols.index("ys")])
            zs = float(parts[cols.index("zs")])
            x = box_lo[0] + xs * (box_hi[0] - box_lo[0])
            y = box_lo[1] + ys * (box_hi[1] - box_lo[1])
            z = box_lo[2] + zs * (box_hi[2] - box_lo[2])
        if wrap:
            x, y, z = wrap_pbc(x, y, z, box_lo, box_hi)
        return x, y, z

    # Parse box from dump for optional update
    dump_box = None
    if update_box:
        dump_box = list(zip(box_lo, box_hi))

    # Detect Atoms section and optional charge/type columns in data file
    # Supported formats: atom-ID atom-type x y z  OR  atom-ID atom-type q x y z
    in_atoms = False
    out_lines = []
    for line in lines:
        stripped = line.strip()

        # Update box bounds
        if update_box and dump_box:
            for dim_i, tag in enumerate(("xlo  xhi", "ylo  yhi", "zlo  zhi")):
                if tag in stripped:
                    lo, hi = dump_box[dim_i]
                    line = f"  {lo:20.8f}  {hi:20.8f}  {tag}\n"
                    break

        if stripped == "Atoms":
            in_atoms = True
            out_lines.append(line)
            continue

        if in_atoms:
            if stripped == "" or stripped.startswith("#"):
                out_lines.append(line)
                continue
            # Check if we've left the Atoms section
            keywords = ("Bonds", "Angles", "Dihedrals", "Impropers",
                        "Velocities", "Masses", "Pair Coeffs",
                        "Bond Coeffs", "Angle Coeffs")
            if any(stripped.startswith(k) for k in keywords):
                in_atoms = False
                out_lines.append(line)
                continue

            parts = stripped.split()
            if not parts:
                out_lines.append(line)
                continue

            atom_id = int(parts[0])
            if atom_id not in frame["atoms"]:
                print(f"Warning: atom {atom_id} not found in dump frame; keeping original.")
                out_lines.append(line)
                continue

            x, y, z = get_coords(frame["atoms"][atom_id])

            # Reconstruct line preserving format (id type [charge] x y z [flags])
            # Detect columns: 2 tokens before coords → type+charge; 1 → type only
            if len(parts) >= 6 and all(_is_float(parts[k]) for k in (2, 3, 4, 5)):
                # id type charge x y z [imx imy imz]
                prefix = f"{parts[0]:>10}  {parts[1]:>3}  {parts[2]:>16}"
                # After wrapping, image flags are all zero
                trailing = parts[6:] if len(parts) > 6 else []
                if wrap and trailing:
                    trailing = ["0"] * len(trailing)
                suffix = "  " + "  ".join(trailing) if trailing else ""
                line = f"{prefix}  {x:18.8f}  {y:18.8f}  {z:18.8f}{suffix}\n"
            else:
                # id type x y z [imx imy imz]
                trailing = parts[5:] if len(parts) > 5 else []
                if wrap and trailing:
                    trailing = ["0"] * len(trailing)
                suffix = "  " + "  ".join(trailing) if trailing else ""
                line = (f"{parts[0]:>10}  {parts[1]:>3}  "
                        f"{x:18.8f}  {y:18.8f}  {z:18.8f}{suffix}\n")

        out_lines.append(line)

    with open(output_file, "w") as f:
        f.writelines(out_lines)

    print(f"Written updated data file to: {output_file}")


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Replace coordinates in a LAMMPS data file from a dump frame.")
    parser.add_argument("data_file", help="Input LAMMPS data file")
    parser.add_argument("dump_file", help="LAMMPS dump file")
    parser.add_argument("--frame", type=int, default=-1,
                        help="Frame index to use (0-based; default: -1 = last frame)")
    parser.add_argument("--output", default=None,
                        help="Output data file (default: <data_file>.new)")
    parser.add_argument("--update-box", action="store_true",
                        help="Also update box dimensions from the dump frame")
    parser.add_argument("--wrap", action="store_true",
                        help="Wrap coordinates into the primary periodic box (PBC)")
    args = parser.parse_args()

    output = args.output or args.data_file.rsplit(".", 1)[0] + "_new.lmp"

    frame = read_dump_frame(args.dump_file, args.frame)
    update_data_file(args.data_file, frame, output, args.update_box, args.wrap)


if __name__ == "__main__":
    main()
