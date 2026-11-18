# -*- coding: utf-8 -*-
"""Tests of the ``cets-dynamo`` command line with the data of a real Dynamo extraction
(``dynamo/tests/data``): 24 particles of the tomogram of TS_03 (Dynamo tomogram 0).

Fixtures in ``dynamo/tests/data``:

- ``crop.tbl``: Dynamo table of the particles (35 columns, tags 1-24, tomogram 0).
- ``particle_000001.em`` ... ``particle_000024.em``: the particles (48 x 48 x 48 voxels;
  their headers have no pixel size).
- ``coordinates_pixel_size.txt``: voxel size (Å) of the particles and of the tomogram they
  were extracted from.
- ``coordinates_0_dynamo_to_cets.yaml``, ``average_0_dynamo_to_cets.yaml``: the CETS
  coordinates and average of the particles, as written by ``dynamo/scripts/example.py``
  (``dynamo_to_cets`` with the voxel size of ``coordinates_pixel_size.txt``).
- ``subtomograms_0_cets_to_dynamo.tbl``: the table written back from them
  (``cets_to_dynamo`` with a tilt range of -60, 60).

The reference outputs are compared with the ones of the command line, so any change of the
conversion is detected; the conversion itself is checked against the table.
"""

import contextlib
import hashlib
import io
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest import TestCase

import numpy as np

from cets_data_model.models.models import Average, PointSet3D, Scale, Translation
from cets_data_model.utils.image_utils import get_em_info
from cets_data_model.utils.yaml_io import load_cets_object
from dynamo.cli import main

DATA_DIR = Path(__file__).parent / "data"
TBL = DATA_DIR / "crop.tbl"
PIXEL_SIZE_FILE = DATA_DIR / "coordinates_pixel_size.txt"
REF_COORDINATES = DATA_DIR / "coordinates_0_dynamo_to_cets.yaml"
REF_AVERAGE = DATA_DIR / "average_0_dynamo_to_cets.yaml"
REF_TBL = DATA_DIR / "subtomograms_0_cets_to_dynamo.tbl"

TOMO_ID = 0
N_PARTICLES = 24
BOX = 48
TILT_RANGE = (-60.0, 60.0)
N_TBL_COLUMNS = 40  # Columns of the tables written by from-cets
# 0-based indices of the table columns
TAG, ALIGNED, AVERAGED = 0, 1, 2
SHIFTS = slice(3, 6)
ANGLES = slice(6, 9)
WEDGE = slice(12, 15)  # ftype, ymintilt, ymaxtilt
TOMO = 19
POSITIONS = slice(23, 26)
WRITTEN_COLUMNS = {TAG, ALIGNED, AVERAGED, TOMO, *range(3, 9), *range(12, 15)}
WRITTEN_COLUMNS |= set(range(23, 26))
ATOL = 1e-6


def read_pixel_size(path: Path = PIXEL_SIZE_FILE) -> float:
    """The pixel size (Å) of a text file holding only that value."""
    return float(path.read_text().strip())


def read_tbl(path: Path) -> np.ndarray:
    """A Dynamo table as a 2-D array (one row per particle)."""
    return np.loadtxt(path, ndmin=2)


def particle_files(directory: Path = DATA_DIR) -> List[Path]:
    return sorted(directory.glob("particle_*.em"))


def tree_digest(directory: Path) -> Dict[str, str]:
    """sha256 of every file of a directory, to check that it is not modified."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def dump_without_paths(obj: PointSet3D | Average) -> Dict[str, Any]:
    """The model as a dict, with the particle paths reduced to their file name (the
    reference files were written from another directory)."""
    data = obj.model_dump(mode="json")
    for particle_map in data.get("particle_maps") or []:
        particle_map["path"] = Path(particle_map["path"]).name
    return data


def run(args: List[str]) -> Tuple[int, str, str]:
    """Runs cets-dynamo with the given arguments; returns the exit code and what it
    printed to stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


class CetsDynamoCliFixturesTest(TestCase):
    pixel_size: float

    @classmethod
    def setUpClass(cls) -> None:
        cls.pixel_size = read_pixel_size()

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="cets_dynamo_cli_fixtures"))
        self.yaml = self.dir / "subtomograms_cets.yaml"
        self.out_tbl = self.dir / "out" / "crop.tbl"

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def assert_run(self, args: List[str], code: int = 0) -> str:
        """Runs cets-dynamo, checks its exit code and returns what it printed to stderr."""
        result, out, err = run(args)
        self.assertEqual(result, code, msg=f"stdout:\n{out}\nstderr:\n{err}")
        return err

    def _to_cets_args(self, *extra: str, particles_dir: Path = DATA_DIR) -> List[str]:
        return [
            "subtomograms",
            "to-cets",
            "--tbl",
            str(TBL),
            "--particles-dir",
            str(particles_dir),
            "--tomo-id",
            str(TOMO_ID),
            "-o",
            str(self.yaml),
            *extra,
        ]

    def _to_cets(self) -> Tuple[PointSet3D, Average]:
        self.assert_run(self._to_cets_args("--pixel-size", str(self.pixel_size)))
        return (
            load_cets_object(self.yaml, "coordinates", PointSet3D),
            load_cets_object(self.yaml, "average", Average),
        )

    def _from_cets_args(self, *extra: str) -> List[str]:
        return [
            "subtomograms",
            "from-cets",
            "-i",
            str(self.yaml),
            "-o",
            str(self.out_tbl),
            *extra,
        ]

    # ------------------------------------------------------------------ fixtures
    def test_fixture_files(self) -> None:
        """The particle files are those of the table: one per tag, all of tomogram 0, and
        without pixel size in their headers (it is given with --pixel-size)."""
        table = read_tbl(TBL)
        self.assertEqual(table.shape[0], N_PARTICLES)
        np.testing.assert_array_equal(table[:, TOMO], TOMO_ID)
        files = particle_files()
        self.assertEqual(
            sorted(int(f.stem.split("_")[-1]) for f in files),
            sorted(table[:, TAG].astype(int).tolist()),
        )
        for em_file in files:
            info = get_em_info(em_file)
            self.assertEqual((info.size_x, info.size_y, info.size_z), (BOX,) * 3)
            self.assertFalse(info.apix_x)
        self.assertGreater(self.pixel_size, 0)

    # ------------------------------------------------------------------ to-cets
    def test_to_cets_matches_reference(self) -> None:
        point_set, average = self._to_cets()
        ref_point_set = load_cets_object(REF_COORDINATES, "coordinates", PointSet3D)
        ref_average = load_cets_object(REF_AVERAGE, "average", Average)
        self.assertEqual(
            dump_without_paths(point_set), dump_without_paths(ref_point_set)
        )
        self.assertEqual(dump_without_paths(average), dump_without_paths(ref_average))

    def test_to_cets_values(self) -> None:
        """The CETS objects against the values of the table."""
        point_set, average = self._to_cets()
        table = read_tbl(TBL)
        self.assertEqual(point_set.source_tomogram_id, str(TOMO_ID))
        # 1-based Dynamo positions -> 0-based CETS array frame
        np.testing.assert_allclose(
            point_set.origin3D or [], table[:, POSITIONS] - 1, atol=ATOL
        )
        maps = average.particle_maps or []
        self.assertEqual([m.coord_index for m in maps], list(range(N_PARTICLES)))
        self.assertTrue(all(m.source_annotation_id == point_set.id for m in maps))
        self.assertTrue(all(m.source_region_id == str(TOMO_ID) for m in maps))
        # Each particle map is the file of its tag
        self.assertEqual(
            [Path(m.path or "").name for m in maps],
            [f"particle_{int(tag):06d}.em" for tag in table[:, TAG]],
        )
        for m, row in zip(maps, table):
            with self.subTest(tag=int(row[TAG])):
                self.assertEqual((m.width, m.height, m.depth), (BOX,) * 3)
                scale, affine, translation = m.coordinate_transformations or []
                assert isinstance(scale, Scale) and isinstance(translation, Translation)
                self.assertEqual(scale.name, "array_to_physical")
                np.testing.assert_allclose(scale.scale, [self.pixel_size] * 3)
                # Shifts in Å
                np.testing.assert_allclose(
                    translation.translation, row[SHIFTS] * self.pixel_size, atol=ATOL
                )
                # The orientation is a rotation
                rotation = np.array(getattr(affine, "affine"))
                np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
                self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_to_cets_needs_pixel_size(self) -> None:
        """The .em headers have no pixel size: without --pixel-size it is an error."""
        err = self.assert_run(self._to_cets_args(), code=1)
        self.assertIn("voxel size", err)
        self.assertFalse(self.yaml.exists())

    def test_to_cets_errors(self) -> None:
        with_pixel_size = ("--pixel-size", str(self.pixel_size))
        # No particles of that tomogram in the table
        args = self._to_cets_args(*with_pixel_size)
        args[args.index("--tomo-id") + 1] = "5"
        self.assertIn("5", self.assert_run(args, code=1))
        # Missing table
        args = self._to_cets_args(*with_pixel_size)
        args[args.index("--tbl") + 1] = str(self.dir / "missing.tbl")
        self.assert_run(args, code=1)
        # Missing particle file: the error names its tag
        particles = self.dir / "particles"
        particles.mkdir()
        for em_file in particle_files()[:-1]:
            shutil.copy(em_file, particles)
        err = self.assert_run(
            self._to_cets_args(*with_pixel_size, particles_dir=particles), code=1
        )
        self.assertIn(f"tag {N_PARTICLES}", err)
        # Directory without particles
        empty = self.dir / "empty"
        empty.mkdir()
        self.assert_run(
            self._to_cets_args(*with_pixel_size, particles_dir=empty), code=1
        )
        self.assertFalse(self.yaml.exists())

    # ------------------------------------------------------------------ from-cets
    def test_from_cets_matches_reference(self) -> None:
        self._to_cets()
        self.assert_run(
            self._from_cets_args("--tilt-range", *(str(t) for t in TILT_RANGE))
        )
        self.assertEqual(self.out_tbl.read_text(), REF_TBL.read_text())

    def test_round_trip(self) -> None:
        """to-cets + from-cets give back the columns of the table CETS holds."""
        self._to_cets()
        self.assert_run(
            self._from_cets_args("--tilt-range", *(str(t) for t in TILT_RANGE))
        )
        original, written = read_tbl(TBL), read_tbl(self.out_tbl)
        self.assertEqual(written.shape, (N_PARTICLES, N_TBL_COLUMNS))
        np.testing.assert_array_equal(written[:, TAG], original[:, TAG])
        np.testing.assert_allclose(written[:, SHIFTS], original[:, SHIFTS], atol=ATOL)
        np.testing.assert_allclose(written[:, ANGLES], original[:, ANGLES], atol=1e-4)
        np.testing.assert_array_equal(written[:, TOMO], TOMO_ID)
        np.testing.assert_allclose(
            written[:, POSITIONS], original[:, POSITIONS], atol=ATOL
        )
        np.testing.assert_array_equal(written[:, [ALIGNED, AVERAGED]], 1)
        np.testing.assert_array_equal(
            written[:, WEDGE], [[1, *TILT_RANGE]] * N_PARTICLES
        )
        # The columns CETS does not hold are 0
        others = [c for c in range(N_TBL_COLUMNS) if c not in WRITTEN_COLUMNS]
        np.testing.assert_array_equal(written[:, others], 0)

    def test_from_cets_options(self) -> None:
        self._to_cets()
        original = read_tbl(TBL)
        # Default: no missing wedge, the tomogram of the coordinates
        self.assert_run(self._from_cets_args())
        written = read_tbl(self.out_tbl)
        np.testing.assert_array_equal(written[:, WEDGE], 0)
        np.testing.assert_array_equal(written[:, TOMO], TOMO_ID)
        # --tomo-id
        self.assert_run(self._from_cets_args("--tomo-id", "9"))
        np.testing.assert_array_equal(read_tbl(self.out_tbl)[:, TOMO], 9)
        # --pixel-size: the shifts (Å) are expressed in pixels of that size
        self.assert_run(self._from_cets_args("--pixel-size", str(self.pixel_size / 2)))
        np.testing.assert_allclose(
            read_tbl(self.out_tbl)[:, SHIFTS], original[:, SHIFTS] * 2, atol=ATOL
        )

    def test_from_cets_errors(self) -> None:
        # A yaml with only the average (no coordinates) cannot be written as a table
        args = self._from_cets_args()
        args[args.index("-i") + 1] = str(REF_AVERAGE)
        self.assert_run(args, code=1)
        self.assertFalse(self.out_tbl.exists())
        # Wrong output extension
        self._to_cets()
        args = self._from_cets_args()
        args[args.index("-o") + 1] = str(self.dir / "crop.txt")
        self.assert_run(args, code=1)
        self.assertFalse((self.dir / "crop.txt").exists())
        # Missing input
        args = self._from_cets_args()
        args[args.index("-i") + 1] = str(self.dir / "missing.yaml")
        self.assert_run(args, code=1)

    # ------------------------------------------------------------------ command line
    def test_input_files_are_not_modified(self) -> None:
        before = tree_digest(DATA_DIR)
        self._to_cets()
        self.assert_run(self._from_cets_args())
        self.assertEqual(tree_digest(DATA_DIR), before)

    def test_invalid_arguments(self) -> None:
        """Usage errors exit with code 2 (argparse) before running any conversion."""
        invalid: List[List[str]] = [
            [],
            ["subtomograms"],
            ["tomogram", "to-cets"],
            # --tomo-id is required
            ["subtomograms", "to-cets", "--tbl", str(TBL), "--particles-dir", "."],
            # --tomo-id is an integer
            self._to_cets_args()[:-2] + ["--tomo-id", "TS_03"],
            ["subtomograms", "from-cets", "-i", str(self.yaml)],  # -o is required
            self._from_cets_args("--tilt-range", "-60"),  # MIN and MAX
        ]
        for args in invalid:
            with self.subTest(args=args):
                with self.assertRaises(SystemExit) as ctx:
                    run(args)
                self.assertEqual(ctx.exception.code, 2)
        self.assertFalse(self.yaml.exists())

    def test_help(self) -> None:
        for args in (
            ["--help"],
            ["subtomograms", "to-cets", "--help"],
            ["subtomograms", "from-cets", "--help"],
        ):
            with self.subTest(args=args):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    with self.assertRaises(SystemExit) as ctx:
                        main(args)
                self.assertEqual(ctx.exception.code, 0)
                self.assertIn("cets-dynamo", out.getvalue())
