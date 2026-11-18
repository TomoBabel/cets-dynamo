# -*- coding: utf-8 -*-
import contextlib
import io
import shutil
import tempfile
from pathlib import Path
from typing import List
from unittest import TestCase

import numpy as np

from cets_data_model.models.models import Average, PointSet3D
from cets_data_model.utils.yaml_io import load_cets_object
from dynamo.cli import main
from dynamo.converters.subtomograms import DynamoSetOfSubtomograms

TOMO_ID = 7
PIXEL_SIZE = 2.5
BOX = 8
# tag, shifts (px), tdrot, tilt, narot (degrees), 1-based positions (px)
PARTICLES = [
    (1, [1.5, -0.5, 2.0], [30.0, 45.0, -60.0], [20.0, 31.0, 42.0]),
    (2, [0.0, 0.25, -1.0], [-120.0, 100.0, 15.0], [101.0, 52.0, 13.0]),
    (3, [-2.0, 1.0, 0.5], [170.0, 10.0, 90.0], [64.0, 64.0, 32.0]),
]


def run(args: List[str]) -> int:
    """Runs cets-dynamo with the given arguments, hiding its messages."""
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return main(args)


def write_em(em_file: Path, box: int) -> None:
    """Minimal .em particle: get_em_info only needs the header."""
    header = bytearray(512)
    header[0], header[3] = 6, 5  # PC byte order, float32
    header[4:16] = np.array([box, box, box], dtype="<i4").tobytes()
    em_file.write_bytes(bytes(header) + np.zeros(box**3, dtype="<f4").tobytes())


class CetsDynamoCliTest(TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="cets_dynamo_cli"))
        self.particles_dir = self.dir / "particles"
        self.particles_dir.mkdir()
        rows = []
        for tag, shifts, angles, position in PARTICLES:
            row = [0.0] * 40
            row[0] = tag
            row[3:6] = shifts
            row[6:9] = angles
            row[19] = TOMO_ID
            row[23:26] = position
            rows.append(
                " ".join(DynamoSetOfSubtomograms._format_tbl_value(v) for v in row)
            )
            write_em(self.particles_dir / f"particle_{tag:06d}.em", BOX)
        self.tbl = self.dir / "crop.tbl"
        self.tbl.write_text("\n".join(rows) + "\n")
        self.yaml = self.dir / "subtomograms_cets.yaml"

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _to_cets(self) -> None:
        code = run(
            [
                "subtomograms",
                "to-cets",
                "--tbl",
                str(self.tbl),
                "--particles-dir",
                str(self.particles_dir),
                "--tomo-id",
                str(TOMO_ID),
                "--pixel-size",
                str(PIXEL_SIZE),
                "-o",
                str(self.yaml),
            ]
        )
        self.assertEqual(code, 0)

    def test_subtomograms_to_cets(self) -> None:
        self._to_cets()
        point_set = load_cets_object(self.yaml, "coordinates", PointSet3D)
        average = load_cets_object(self.yaml, "average", Average)
        self.assertEqual(point_set.source_tomogram_id, str(TOMO_ID))
        # 1-based Dynamo positions -> 0-based CETS array frame
        np.testing.assert_allclose(
            point_set.origin3D, [np.array(p[3]) - 1 for p in PARTICLES]
        )
        particle_maps = average.particle_maps or []
        self.assertEqual(len(particle_maps), len(PARTICLES))
        self.assertEqual([pm.coord_index for pm in particle_maps], [0, 1, 2])
        self.assertEqual(particle_maps[0].width, BOX)
        # Shifts in Å
        np.testing.assert_allclose(
            particle_maps[0].coordinate_transformations[2].translation,
            np.array(PARTICLES[0][1]) * PIXEL_SIZE,
        )

    def test_subtomograms_round_trip(self) -> None:
        self._to_cets()
        out = self.dir / "out" / "sub" / "crop.tbl"  # dirs created by the command
        code = run(
            [
                "subtomograms",
                "from-cets",
                "-i",
                str(self.yaml),
                "-o",
                str(out),
                "--tilt-range",
                "-60",
                "60",
            ]
        )
        self.assertEqual(code, 0)
        original = np.loadtxt(self.tbl, ndmin=2)
        written = np.loadtxt(out, ndmin=2)
        self.assertEqual(written.shape, (len(PARTICLES), 40))
        np.testing.assert_allclose(written[:, 0], original[:, 0])  # tags
        np.testing.assert_allclose(written[:, 3:9], original[:, 3:9], atol=1e-4)
        np.testing.assert_allclose(written[:, 19], TOMO_ID)
        np.testing.assert_allclose(written[:, 23:26], original[:, 23:26], atol=1e-6)
        # Missing wedge from --tilt-range
        np.testing.assert_allclose(written[:, 12:15], [[1, -60, 60]] * len(PARTICLES))

    def test_from_cets_tomo_id(self) -> None:
        self._to_cets()
        out = self.dir / "crop_9.tbl"
        code = run(
            [
                "subtomograms",
                "from-cets",
                "-i",
                str(self.yaml),
                "-o",
                str(out),
                "--tomo-id",
                "9",
            ]
        )
        self.assertEqual(code, 0)
        written = np.loadtxt(out, ndmin=2)
        np.testing.assert_allclose(written[:, 19], 9)
        np.testing.assert_allclose(written[:, 12:15], 0)  # no missing wedge

    def test_errors_give_exit_code_1(self) -> None:
        # No particles of that tomogram
        no_particles = [
            "subtomograms",
            "to-cets",
            "--tbl",
            str(self.tbl),
            "--particles-dir",
            str(self.particles_dir),
            "--tomo-id",
            "3",
            "-o",
            str(self.yaml),
        ]
        self.assertEqual(run(no_particles), 1)
        # Missing table
        missing_tbl = list(no_particles)
        missing_tbl[3] = str(self.dir / "missing.tbl")
        missing_tbl[7] = str(TOMO_ID)
        self.assertEqual(run(missing_tbl), 1)
        # A yaml without the average cannot be written as a table
        self.yaml.write_text("coordinates: {id: c, annotation_type: point_set_3D}\n")
        no_average = [
            "subtomograms",
            "from-cets",
            "-i",
            str(self.yaml),
            "-o",
            str(self.dir / "x.tbl"),
        ]
        self.assertEqual(run(no_average), 1)
        # Wrong output extension
        self._to_cets()
        wrong_ext = [
            "subtomograms",
            "from-cets",
            "-i",
            str(self.yaml),
            "-o",
            str(self.dir / "x.txt"),
        ]
        self.assertEqual(run(wrong_ext), 1)
