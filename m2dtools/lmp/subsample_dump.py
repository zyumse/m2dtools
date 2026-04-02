"""
Subsample a LAMMPS dump file by start frame index and frequency.

Streams the input dump and writes only frames with index in
[start, start+freq, start+2*freq, ...]. No need to load the full dump into memory.
"""

import argparse
import sys


def subsample_dump(in_path, out_path, start=0, freq=1):
    """
    Write a smaller dump file with frames at indices start, start+freq, start+2*freq, ...

    Parameters
    ----------
    in_path : str
        Path to the input LAMMPS dump file.
    out_path : str
        Path to the output dump file.
    start : int, default 0
        Index of the first frame to include (0-based).
    freq : int, default 1
        Take every freq-th frame after start.

    Returns
    -------
    int
        Number of frames written.
    """
    if freq < 1:
        raise ValueError("freq must be >= 1")
    if start < 0:
        raise ValueError("start must be >= 0")

    frames_written = 0
    frame_index = -1
    buffer = []

    with open(in_path, "r") as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            if line.startswith("ITEM: TIMESTEP"):
                if frame_index >= 0 and buffer:
                    if frame_index >= start and (frame_index - start) % freq == 0:
                        f_out.writelines(buffer)
                        frames_written += 1
                    buffer = []
                frame_index += 1

            buffer.append(line)

        # flush last frame
        if buffer and frame_index >= 0:
            if frame_index >= start and (frame_index - start) % freq == 0:
                f_out.writelines(buffer)
                frames_written += 1

    return frames_written


def main():
    parser = argparse.ArgumentParser(
        description="Subsample a LAMMPS dump file by start and frequency."
    )
    parser.add_argument(
        "input_dump",
        help="Path to the input LAMMPS dump file.",
    )
    parser.add_argument(
        "output_dump",
        help="Path to the output (subsampled) dump file.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        metavar="N",
        help="Index of the first frame to include (0-based). Default: 0.",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=1,
        metavar="N",
        help="Take every N-th frame after start. Default: 1 (all frames after start).",
    )
    args = parser.parse_args()

    try:
        n = subsample_dump(
            args.input_dump,
            args.output_dump,
            start=args.start,
            freq=args.freq,
        )
    except FileNotFoundError:
        print(f"Error: input file not found: {args.input_dump}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {n} frames to {args.output_dump}")


if __name__ == "__main__":
    main()
