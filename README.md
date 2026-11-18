# cets-dynamo

Converters between [Dynamo](https://wiki.dynamo.biozentrum.unibas.ch/) subtomogram data and
CETS, the cryo-electron tomography metadata standard defined in
[cets-data-models](https://github.com/TomoBabel/cets-data-models).

| Data | Dynamo files | CETS objects | Directions |
|---|---|---|---|
| Subtomograms | table (`.tbl`) and particle files (`particle_<tag>.em`) | `PointSet3D` (the particle positions in the tomogram) and `Average` with one `ParticleMap` per particle (its file and pose) | Dynamo → CETS, CETS → Dynamo (table) |

## Installation

Python 3.11 or newer. The CETS data model is not declared as a dependency, so install it first:

```bash
pip install -e /path/to/cets-data-models
pip install -e .          # or: pip install -e .[dev] for the development tools
```

This installs the `dynamo` package and the `cets-dynamo` command.

## Command line

```text
cets-dynamo subtomograms {to-cets,from-cets} [options]
```

```bash
# Dynamo -> CETS: the particles of tomogram 1 of the table
cets-dynamo subtomograms to-cets --tbl crop.tbl --particles-dir particles \
    --tomo-id 1 --pixel-size 4.4 -o subtomograms_1_cets.yaml

# CETS -> Dynamo table
cets-dynamo subtomograms from-cets -i subtomograms_1_cets.yaml -o out/crop.tbl --tilt-range -60 60
```

`to-cets` writes a CETS yaml file with the entries `coordinates` (`PointSet3D`) and `average`
(`Average`); `from-cets` reads it. Every sub-command has `--help`. Errors print a one-line
message and exit with code 1; put `--debug` before the data type to also print the traceback.

| `to-cets` option | Meaning |
|---|---|
| `--tbl` (required) | Dynamo table. |
| `--particles-dir` (required) | Directory with the particle files, named after their tag (table column 1): `particle_<tag>.em`, zero-padded or not. |
| `--tomo-id` (required) | Dynamo numeric id of the tomogram (column 20) whose particles are converted. |
| `--pixel-size` | Voxel size of the particles (Å). Dynamo does not record it: default, the `.em` header; an error is raised if it does not have it. |

| `from-cets` option | Meaning |
|---|---|
| `-i` (required) | CETS yaml with the coordinates and the average. |
| `-o` (required) | Table to write. |
| `--tomo-id` | Tomogram number (column 20). Default: the `source_tomogram_id` of the coordinates, which must then be an integer. |
| `--pixel-size` | Voxel size (Å) to express the shifts in pixels. Default: the `array_to_physical` scale of each particle. |
| `--tilt-range MIN MAX` | Tilt range of the missing wedge (columns 13-15). Without it, the table declares no missing wedge. |

## Python API

```python
from dynamo.converters.subtomograms import DynamoSetOfSubtomograms

converter = DynamoSetOfSubtomograms()
point_set, average = converter.dynamo_to_cets("crop.tbl", "particles", tomo_id=1, pixel_size=4.4)
converter.cets_to_dynamo(point_set, average, "out/crop.tbl", tomo_id=1)
```

## Conventions

- **Particle files** are matched with the rows of the table by their tag (column 1): the number
  at the end of the file name. The order of the table and of the file names does not matter,
  and the particles of other tomograms of the table need no files. A missing file or two files
  with the same tag are errors. When writing, the tag is taken back from the file names
  (`particle_<tag>.em`) if all of them follow that pattern and the tags are unique; otherwise
  the rows are numbered 1..N.
- **Positions.** Dynamo positions (columns 24-26) are 1-based MATLAB voxel indices
  (`dynamo_crop` centres the box of position `p` on the voxels `p-L/2 .. p+L/2-1`); CETS
  positions (`PointSet3D.origin3D`) are 0-based voxel positions in the tomogram, so 1 is
  subtracted when reading and added when writing.
- **Pose.** Each `ParticleMap` holds, after its `array_to_physical` scale, an `Affine` and a
  `Translation` applied in that order: a point `r` of the reference lands on `p = R @ r + t` in
  the particle box (physical frame, Å). `R` is the transpose of `dynamo_euler2matrix` for the
  (tdrot, tilt, narot) of the table and `t` the table shifts (columns 4-6) in Å. It is the
  inverse of the matrix Scipion stores, and the same pose cets-scipion gives for the particle.
- **Pixel size** is required: Dynamo files do not record it reliably.

## Limitations

- Only the columns above are converted: the scores, classes, flags and other columns of the
  table are not kept, and the missing wedge is only written (`--tilt-range`), not read.
- The `PointSet3D` has no `array_to_physical` scale, as the tomogram voxel size is not given;
  its `source_tomogram_id` is Dynamo's tomogram number.
- The orientation is stored on the `ParticleMap`s, not on the coordinates.

## Development

```bash
pip install -e .[dev]
pre-commit install
pytest dynamo/tests/tests_*.py    # the test files must be given: they are named tests_*.py
ruff check . && ruff format .
mypy
```

An example script is in `dynamo/scripts/`.

### Tests

| File | What it tests |
|---|---|
| `dynamo/tests/tests_subtomograms.py` | The matching of the particle files with the rows of the table by their tag, on small synthetic tables. |
| `dynamo/tests/tests_cli.py` | Both `cets-dynamo subtomograms` sub-commands, on a small synthetic table and particles (built in a temporary directory). |
| `dynamo/tests/tests_cli_fixtures.py` | Both sub-commands on the particles of a real Dynamo extraction (`dynamo/tests/data`): the outputs are compared with reference outputs and with the values of the table, plus the round trip, the options and the errors. |

```bash
pytest dynamo/tests/tests_cli_fixtures.py                 # one file
pytest dynamo/tests/tests_cli_fixtures.py -k from_cets    # some tests
```

The test data (`dynamo/tests/data`) holds 24 particles extracted by Dynamo from the tomogram of
TS_03 (Dynamo tomogram `0`):

| File | Content |
|---|---|
| `crop.tbl` | Dynamo table of the particles (35 columns, tags 1-24, tomogram 0). |
| `particle_000001.em` ... `particle_000024.em` | The particles (48 x 48 x 48 voxels; their headers have no pixel size). |
| `coordinates_pixel_size.txt` | Voxel size of the particles (10.8 Å/voxel), given with `--pixel-size`. |
| `coordinates_0_dynamo_to_cets.yaml`, `average_0_dynamo_to_cets.yaml` | Reference CETS coordinates and average of the particles: `dynamo_to_cets` of the files above with 10.8 Å/voxel, written with `write_subtomograms_yaml`. |
| `subtomograms_0_cets_to_dynamo.tbl` | Reference table written back from them (`cets_to_dynamo` with a tilt range of -60, 60). |


