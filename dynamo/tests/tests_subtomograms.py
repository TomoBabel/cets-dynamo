# -*- coding: utf-8 -*-
import shutil
import tempfile
from pathlib import Path
from typing import List
from unittest import TestCase

from dynamo.converters.subtomograms import DynamoSetOfSubtomograms
from dynamo.tests.tests_cli import write_em

TOMO_ID = 7
OTHER_TOMO_ID = 8


def tbl_row(tag: int, tomo_id: int = TOMO_ID) -> str:
    row = [0.0] * 40
    row[0] = tag
    row[19] = tomo_id
    row[23:26] = [10.0 + tag, 20.0, 30.0]
    return " ".join(DynamoSetOfSubtomograms._format_tbl_value(v) for v in row)


class DynamoParticleFilesTest(TestCase):
    """The particles of the table are matched with their files by their tag."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="cets_dynamo_tags"))
        self.particles_dir = self.dir / "particles"
        self.particles_dir.mkdir()
        self.tbl = self.dir / "crop.tbl"

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _convert(self, rows: List[str]) -> List[str]:
        self.tbl.write_text("\n".join(rows) + "\n")
        result = DynamoSetOfSubtomograms().dynamo_to_cets(
            self.tbl, self.particles_dir, tomo_id=TOMO_ID, pixel_size=1.0
        )
        assert result is not None
        point_set, average = result
        return [Path(pm.path).name for pm in average.particle_maps]

    def test_unpadded_names_and_unsorted_table(self) -> None:
        # Sorted by name, particle_10.em comes before particle_2.em: the files must be
        # matched by tag, not by position. The box size identifies each file.
        write_em(self.particles_dir / "particle_2.em", 4)
        write_em(self.particles_dir / "particle_10.em", 6)
        write_em(self.particles_dir / "particle_000003.em", 8)
        names = self._convert([tbl_row(10), tbl_row(3), tbl_row(2)])
        self.assertEqual(
            names, ["particle_10.em", "particle_000003.em", "particle_2.em"]
        )

    def test_box_size_of_each_particle(self) -> None:
        write_em(self.particles_dir / "particle_2.em", 4)
        write_em(self.particles_dir / "particle_10.em", 6)
        self.tbl.write_text("\n".join([tbl_row(10), tbl_row(2)]) + "\n")
        result = DynamoSetOfSubtomograms().dynamo_to_cets(
            self.tbl, self.particles_dir, tomo_id=TOMO_ID, pixel_size=1.0
        )
        assert result is not None
        self.assertEqual([pm.width for pm in result[1].particle_maps], [6, 4])

    def test_other_tomograms_need_no_files(self) -> None:
        # A table with the particles of several tomograms, converting one of them
        write_em(self.particles_dir / "particle_1.em", 4)
        names = self._convert([tbl_row(1), tbl_row(2, OTHER_TOMO_ID)])
        self.assertEqual(names, ["particle_1.em"])

    def test_missing_particle_file(self) -> None:
        write_em(self.particles_dir / "particle_1.em", 4)
        with self.assertRaisesRegex(FileNotFoundError, "tag 2"):
            self._convert([tbl_row(1), tbl_row(2)])

    def test_duplicated_tag(self) -> None:
        write_em(self.particles_dir / "particle_7.em", 4)
        write_em(self.particles_dir / "particle_007.em", 4)
        with self.assertRaisesRegex(ValueError, "tag 7"):
            self._convert([tbl_row(7)])
