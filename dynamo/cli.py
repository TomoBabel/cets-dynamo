"""Command line of the Dynamo <-> CETS converters (``cets-dynamo``).

Examples::

    cets-dynamo subtomograms to-cets --tbl crop.tbl --particles-dir particles \\
        --tomo-id 1 --pixel-size 4.4 -o subtomograms_1_cets.yaml
    cets-dynamo subtomograms from-cets -i subtomograms_1_cets.yaml -o out/crop.tbl \\
        --tilt-range -60 60

``to-cets`` writes a CETS yaml file with one entry per kind of object (``coordinates``,
the picked coordinates, and ``average``, the particle maps), which is what ``from-cets``
reads.
"""

import argparse
from pathlib import Path
from typing import List, Optional

from cets_data_model.models.models import Average, PointSet3D
from cets_data_model.utils.cli import (
    FROM_CETS,
    TO_CETS,
    add_data_type,
    add_input_yaml,
    add_output_yaml,
    check_outputs,
    prepare_outputs,
    run_cli,
)
from cets_data_model.utils.yaml_io import load_cets_object, write_cets_yaml
from dynamo.converters.subtomograms import DynamoSetOfSubtomograms

COORDINATES_KEY = "coordinates"
AVERAGE_KEY = "average"


# ------------------------------------------------------------------------- subtomograms
def subtomograms_to_cets(args: argparse.Namespace) -> None:
    result = DynamoSetOfSubtomograms().dynamo_to_cets(
        args.tbl,
        args.particles_dir,
        args.tomo_id,
        pixel_size=args.pixel_size,
    )
    if result is None:
        raise ValueError(f"No subtomograms of the tomogram {args.tomo_id} were found.")
    point_set, average = result
    out = write_cets_yaml(
        args.output, {COORDINATES_KEY: point_set, AVERAGE_KEY: average}
    )
    print(f"CETS subtomograms written -> {out}")


def subtomograms_from_cets(args: argparse.Namespace) -> None:
    point_set = load_cets_object(args.input, COORDINATES_KEY, PointSet3D)
    average = load_cets_object(args.input, AVERAGE_KEY, Average)
    outputs = prepare_outputs(args.output)
    DynamoSetOfSubtomograms().cets_to_dynamo(
        point_set,
        average,
        args.output,
        tomo_id=args.tomo_id,
        pixel_size=args.pixel_size,
        tilt_range=(
            (args.tilt_range[0], args.tilt_range[1]) if args.tilt_range else None
        ),
    )
    check_outputs(outputs)


def _add_subtomograms(
    data_types: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    directions = add_data_type(
        data_types,
        "subtomograms",
        "Subtomograms: table (.tbl) and particle files (.em).",
    )

    to_cets = directions.add_parser(
        TO_CETS,
        help="Dynamo table + particles -> CETS yaml (coordinates and average).",
    )
    to_cets.add_argument("--tbl", required=True, type=Path, help="Dynamo table (.tbl).")
    to_cets.add_argument(
        "--particles-dir",
        required=True,
        type=Path,
        help="Directory with the particle files (.em), named after their tag (table "
        "column 1): particle_<tag>.em, zero-padded or not.",
    )
    to_cets.add_argument(
        "--tomo-id",
        required=True,
        type=int,
        help="Dynamo numeric identifier of the tomogram (column 20) whose subtomograms "
        "are converted.",
    )
    to_cets.add_argument(
        "--pixel-size",
        type=float,
        help="Voxel size of the particles (Å/px). Dynamo does not record it. Default: "
        "the .em header; an error is raised if it does not have it.",
    )
    add_output_yaml(to_cets)
    to_cets.set_defaults(func=subtomograms_to_cets)

    from_cets = directions.add_parser(
        FROM_CETS, help="CETS yaml -> Dynamo table (.tbl)."
    )
    add_input_yaml(
        from_cets, "CETS yaml with the coordinates and the average (particle maps)."
    )
    from_cets.add_argument(
        "-o", "--output", required=True, type=Path, help="Dynamo table to write (.tbl)."
    )
    from_cets.add_argument(
        "--tomo-id",
        type=int,
        help="Dynamo numeric identifier of the tomogram (column 20). Default: the "
        "source tomogram id of the coordinates, which must then be an integer.",
    )
    from_cets.add_argument(
        "--pixel-size",
        type=float,
        help="Voxel size of the particles (Å/px), to express the shifts in pixels. "
        "Default: the array_to_physical scale of each particle.",
    )
    from_cets.add_argument(
        "--tilt-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Minimum and maximum tilt angles (degrees, around Y) of the missing wedge. "
        "Without it, the table declares no missing wedge.",
    )
    from_cets.set_defaults(func=subtomograms_from_cets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cets-dynamo",
        description="Converts Dynamo files to and from CETS metadata (yaml).",
    )
    data_types = parser.add_subparsers(
        dest="data_type", metavar="{subtomograms}", required=True
    )
    _add_subtomograms(data_types)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(build_parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
