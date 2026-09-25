import os
from pathlib import Path
from typing import List, Tuple

import yaml

from cets_data_model.models.models import (
    CoordinateSystem,
    Axis,
    AxisType,
    Scale,
    PointSet3D,
    Average,
)
from dynamo.constants import EM_EXT


def write_subtomograms_yaml(
    point_set: PointSet3D,
    average: Average,
    tomo_id: int | str,
    output_directory: Path,
) -> None:
    """Writes the (PointSet3D, Average) pair produced by ``dynamo_to_cets`` to yaml.

    The coordinates annotation and the average are written to separate files, mirroring the
    fact that in the data model they live in different containers (Region.annotations and
    Dataset.averages, respectively).
    """
    _write_obj_yaml(
        point_set, output_directory / f"coordinates_{tomo_id}_dynamo_to_cets.yaml"
    )
    _write_obj_yaml(
        average, output_directory / f"average_{tomo_id}_dynamo_to_cets.yaml"
    )


def _write_obj_yaml(cets_obj: PointSet3D | Average, yaml_file: Path) -> None:
    metadata_dict = cets_obj.model_dump(mode="json")
    with open(yaml_file, "w") as f:
        yaml.dump(metadata_dict, f, sort_keys=False, explicit_start=True)
    print(f"yaml file successfully written! -> {yaml_file}")


def gen_coordinate_systems(
    name: str, ndim: int = 2
) -> Tuple[CoordinateSystem, CoordinateSystem]:
    """Builds the (array, physical) coordinate-system pair for an ``ndim`` image/frame.

    Names follow the CETS proposal convention ``{name}_array`` / ``{name}_physical``. The array
    system is pixel/array coords (unitless); the physical system is in Å. ``ndim`` is 2 for 2-D
    images (x, y) and 3 for volumes (x, y, z).
    """
    axes = ("x", "y", "z")[:ndim]
    array_cs = CoordinateSystem(
        name=f"{name}_array",
        axes=[Axis(name=a, axis_type=AxisType.array, axis_unit=None) for a in axes],
    )
    physical_cs = CoordinateSystem(
        name=f"{name}_physical",
        axes=[
            Axis(name=a, axis_type=AxisType.space, axis_unit="angstrom") for a in axes
        ],
    )
    return array_cs, physical_cs


def gen_array_to_physical(
    pixel_size: float, array_cs_name: str, physical_cs_name: str, ndim: int = 2
) -> Scale:
    """The single canonical ``array_to_physical`` transformation for an image: a Scale mapping
    pixel/array coordinates to physical (Å) coordinates by the (isotropic) pixel/voxel size.
    The spec requires exactly one such transformation per image."""
    return Scale(
        scale=[pixel_size] * ndim,
        name="array_to_physical",
        input=array_cs_name,
        output=physical_cs_name,
    )


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
