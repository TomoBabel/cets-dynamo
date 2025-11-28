import os
from pathlib import Path
from typing import List

from dynamo.constants import EM_EXT


def validate_file(filename: os.PathLike, expected_ext: str) -> Path:
    if filename is None:
        raise ValueError("The introduced file cannot be None")
    p = Path(filename).expanduser()
    try:
        p = p.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"{filename} does not exist: {p}") from None

    if not p.is_file():
        raise IsADirectoryError(f"{filename} must be a file, not a directory: {p}")

    if not os.access(p, os.R_OK):
        raise PermissionError(f"No read permission for {filename}: {p}")

    ext = p.suffix
    if ext != expected_ext:
        raise ValueError(f"Invalid file extension '{ext}'. Expected: {expected_ext}")
    return p


def num_particles_in_tbl(tbl_file: os.PathLike) -> int:
    with open(tbl_file, "r") as f:
        return sum(1 for _ in f)


def get_particle_files(particles_dir: os.PathLike) -> List[str]:
    particles_dir = Path(particles_dir)
    pattern = f"*{EM_EXT}"
    particle_files = [str(em_file) for em_file in particles_dir.glob(pattern)]
    if not particle_files:
        raise Exception("No Dynamo particles (.em) were found in the provided path.")
    return particle_files
