import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

from cets_data_model.models.models import (
    PointSet3D,
    ParticleMap,
    Average,
    AnnotationType,
    Affine,
    Scale,
    Translation,
)
from cets_data_model.utils.image_utils import get_em_info
from dynamo.constants import TBL_EXT, TBL_N_COLUMNS, TBL_POSITION_OFFSET
from dynamo.utils.utils import (
    validate_file,
    get_particle_files_by_tag,
    gen_coordinate_systems,
    gen_array_to_physical,
)


class DynamoSetOfSubtomograms:
    def dynamo_to_cets(
        self,
        tbl_file: os.PathLike,
        dynamo_particles_directory: os.PathLike,
        tomo_id: int,
        pixel_size: float | None = None,
    ) -> Tuple[PointSet3D, Average] | None:
        """Converts a set of subtomograms in Dynamo tbl format corresponding to the
        introduced tomogram identifier into CETS metadata.

        In the new data model the old ``Particle3D``/``Particle3DSet`` (which fused a picked
        coordinate and an extracted subvolume into one object) is split into two entities
        (Option A):

        * the picked coordinates -> a ``PointSet3D`` annotation (stored under
          ``Region.annotations``), linked to its tomogram via ``source_tomogram_id``;
        * each extracted subvolume -> a ``ParticleMap`` (stored under ``Average.particle_maps``),
          linked back to a single picked coordinate via ``source_region_id`` +
          ``source_annotation_id`` (the region and the ``PointSet3D`` it lives in) plus
          ``coord_index`` (the 0-based index into ``PointSet3D.origin3D``, so it must stay
          aligned with the order in which the coordinates are appended below).

        Dynamo positions (columns 24-26) are 1-based voxel indices; they are converted to the
        0-based array frame of the tomogram.

        The pose of each particle is stored in its ``coordinate_transformations`` after the
        ``array_to_physical`` scale, as an ``Affine`` followed by a ``Translation``, applied in
        that order: a point r of the reference lands on p = R @ r + t in the particle box
        (physical frame, Å), R being the transpose of Dynamo's ``dynamo_euler2matrix`` for the
        (tdrot, tilt, narot) of the table and t the table shifts. It is the inverse of the
        matrix Scipion stores for the same particle (particle -> reference).

        :param tbl_file: path to the tbl file containing the subtomograms data.
        :type tbl_file: os.PathLike.

        :param dynamo_particles_directory: path to the directory containing the
        subtomograms binary files (.em).
        :type dynamo_particles_directory: os.PathLike.

        :param tomo_id: Dynamo numeric identifier of the tomogram. It is used to indicate
        the tomogram from which the subtomograms will be converted, as in Dynamo the
        subtomograms from all the tomograms are stored together.
        :type tomo_id: int.

        :param pixel_size: the subtomogram box voxel size in Å, used for each ParticleMap's
        array_to_physical scale and to express the pose shifts in Å. Dynamo's .em particle
        headers and the .tbl do not record a pixel size, so it must be supplied by the caller;
        when omitted it is read from the .em header (usually absent); if it is not
        there either, a ValueError is raised.
        :type pixel_size: float, optional. Defaults to None.

        Each particle of the table is matched with its file by its tag (column 1): Dynamo
        names the particle files after it (particle_<tag>.em, with or without zero padding).
        The order of the table and of the file names does not matter.
        """
        # Validate the tbl file
        validate_file(tbl_file, expected_ext=TBL_EXT)
        # Files of the particles directory, by the tag in their names
        particle_files_by_tag = get_particle_files_by_tag(dynamo_particles_directory)

        # TODO (open question #3): id-generation policy. tomo_id is reused as the
        # Region id and as the seed of the annotation id. These must be unique within
        # their respective scopes (Region within Dataset, Annotation within
        # Region.annotations).
        region_id = str(tomo_id)
        annotation_id = f"dynamo_coords_{tomo_id}"

        with open(tbl_file, "r") as dynamo_tbl:
            origin_3d: List[List[float]] = []
            particle_maps: List[ParticleMap] = []
            for line in dynamo_tbl:
                parts = line.split()
                if not parts:
                    continue
                vol_id = int(parts[19])
                if vol_id != tomo_id:
                    continue
                tag = int(float(parts[0]))
                particle_fn = particle_files_by_tag.get(tag)
                if particle_fn is None:
                    raise FileNotFoundError(
                        f"No file of the particle with tag {tag} (particle_{tag}.em, "
                        f"with or without zero padding) in {dynamo_particles_directory}."
                    )
                em_info = get_em_info(particle_fn)
                # Voxel size (Å): the one given or, if not given, the one of the .em header.
                # Dynamo .em files usually carry no pixel size, so it should be provided.
                voxel_size = pixel_size or em_info.apix_x
                if not voxel_size:
                    raise ValueError(
                        f"The voxel size of the particles is unknown: the header of "
                        f"{particle_fn} does not have it, so pixel_size must be provided."
                    )
                # 1-based Dynamo positions -> 0-based CETS array frame
                origin_3d.append(
                    [float(coord) - TBL_POSITION_OFFSET for coord in parts[23:26]]
                )
                # Every image (particle box) gets an array (voxel, unitless) and a physical
                # (Å) coordinate system plus exactly one canonical array_to_physical scale
                # (the box voxel size), per the spec.
                particle_name = f"particle_{len(particle_maps):03d}"
                array_cs, physical_cs = gen_coordinate_systems(particle_name, ndim=3)
                array_to_physical = gen_array_to_physical(
                    voxel_size, array_cs.name, physical_cs.name, ndim=3
                )
                particle_maps.append(
                    ParticleMap(
                        path=particle_fn,
                        width=em_info.size_x,
                        height=em_info.size_y,
                        depth=em_info.size_z,
                        # Link this subvolume back to the picked coordinate it was extracted
                        # from: region + PointSet3D annotation + point index.
                        source_region_id=region_id,
                        source_annotation_id=annotation_id,
                        coord_index=len(particle_maps),
                        coordinate_systems=[array_cs, physical_cs],
                        # Pose: rotation first, then translation (p = R @ r + t)
                        coordinate_transformations=[
                            array_to_physical,
                            self._get_particle_transform(parts, physical_cs.name),
                            self._get_particle_translation(
                                parts, physical_cs.name, voxel_size
                            ),
                        ],
                    )
                )
            if not particle_maps:
                raise Exception(
                    f"No particle files were found matching the introduced Dynamo's "
                    f"tomogram numeric identifier [{tomo_id}]."
                )

            # Picked coordinates are positions in the tomogram's array (voxel) frame.
            tomo_array_cs, _ = gen_coordinate_systems(str(tomo_id), ndim=3)
            point_set = PointSet3D(
                id=annotation_id,
                name=f"Dynamo coordinates for {tomo_id}",
                annotation_type=AnnotationType.point_set_3D,
                source_tomogram_id=str(tomo_id),
                origin3D=origin_3d,
                coordinate_systems=[tomo_array_cs],
            )
            average = Average(
                name=f"Dynamo subtomograms for {tomo_id}",
                particle_maps=particle_maps,
            )
            return point_set, average

    def cets_to_dynamo(
        self,
        point_set: PointSet3D,
        average: Average,
        tbl_file: os.PathLike,
        tomo_id: int | None = None,
        pixel_size: float | None = None,
        tilt_range: Tuple[float, float] | None = None,
    ) -> Path:
        """Writes a Dynamo table (.tbl) from the (PointSet3D, Average) pair produced by
        ``dynamo_to_cets``: one row per ``ParticleMap`` of the average, in the order they
        are stored.

        Only the columns CETS holds are filled (1-based Dynamo column numbers):

        * 1 (tag): taken from the particle file name (``particle_000012.em`` -> 12) when
          every particle follows that pattern and the tags are unique; otherwise the rows
          are numbered 1..N.
        * 2, 3 (aligned, averaged): always 1.
        * 4-6 (shifts, pixels): the particle ``Translation`` (Å) divided by the voxel size.
        * 7-9 (tdrot, tilt, narot, degrees): from the particle ``Affine``. When tilt is 0
          or 180 only tdrot + narot (or their difference) is defined, so tdrot is written
          as 0: the triplet may differ from the original one, the rotation does not.
        * 13-15 (ftype, ymintilt, ymaxtilt): missing wedge, only if ``tilt_range`` is
          given, as CETS does not store it.
        * 20 (tomo): ``tomo_id``.
        * 24-26 (x, y, z, pixels): the picked coordinate the particle is linked to, converted
          from the 0-based CETS array frame to Dynamo's 1-based positions.

        Everything else (cc, class, reference, ...) is not stored in CETS and is written
        as 0.

        :param point_set: the picked coordinates.
        :type point_set: PointSet3D.

        :param average: the particle maps, each linked to a coordinate of ``point_set``
        through ``coord_index``.
        :type average: Average.

        :param tbl_file: path of the table to be written.
        :type tbl_file: os.PathLike.

        :param tomo_id: Dynamo numeric identifier of the tomogram (column 20). When
        omitted, it is read from ``point_set.source_tomogram_id``, which must then be
        an integer.
        :type tomo_id: int, optional. Defaults to None.

        :param pixel_size: the subtomogram box voxel size in Å, used to express the
        shifts in pixels. When omitted, it is read from each particle's
        ``array_to_physical`` scale.
        :type pixel_size: float, optional. Defaults to None.

        :param tilt_range: (minimum, maximum) tilt angle in degrees around the Y axis,
        describing the missing wedge. When omitted, the table declares no missing wedge.
        :type tilt_range: Tuple[float, float], optional. Defaults to None.
        """
        tbl_path = Path(tbl_file).expanduser()
        if tbl_path.suffix != TBL_EXT:
            raise ValueError(
                f"Invalid file extension '{tbl_path.suffix}'. Expected: {TBL_EXT}"
            )
        particle_maps = average.particle_maps or []
        if not particle_maps:
            raise ValueError("The introduced average does not contain particle maps.")
        origin_3d = point_set.origin3D or []
        if tomo_id is None:
            try:
                tomo_id = int(str(point_set.source_tomogram_id))
            except ValueError:
                raise ValueError(
                    f"The tomogram id [{point_set.source_tomogram_id}] of the point set "
                    f"is not a Dynamo numeric identifier. Please provide tomo_id."
                ) from None

        tags = self._get_particle_tags(particle_maps)
        rows: List[str] = []
        for ind, (tag, particle_map) in enumerate(zip(tags, particle_maps)):
            if particle_map.source_annotation_id not in (None, point_set.id):
                raise ValueError(
                    f"Particle map {ind} is linked to the annotation "
                    f"[{particle_map.source_annotation_id}], not to the introduced "
                    f"point set [{point_set.id}]."
                )
            coord_index = particle_map.coord_index
            if coord_index is None or not 0 <= coord_index < len(origin_3d):
                raise ValueError(
                    f"Particle map {ind} has no valid coord_index [{coord_index}] for a "
                    f"point set of {len(origin_3d)} coordinates."
                )
            transformations = particle_map.coordinate_transformations or []
            voxel_size = pixel_size or self._get_voxel_size(transformations, ind)
            shifts = [0.0, 0.0, 0.0]
            angles = (0.0, 0.0, 0.0)
            for transformation in transformations:
                if isinstance(transformation, Translation):
                    shifts = [s / voxel_size for s in transformation.translation or []]
                elif isinstance(transformation, Affine) and transformation.affine:
                    angles = self._get_dynamo_euler_angles(
                        np.array(transformation.affine)
                    )
            if len(shifts) != 3:
                raise ValueError(f"Particle map {ind} has no valid 3D translation.")

            columns = [0.0] * TBL_N_COLUMNS
            columns[0] = tag
            columns[1] = 1  # aligned
            columns[2] = 1  # averaged
            columns[3:6] = shifts
            columns[6:9] = angles
            if tilt_range is not None:
                columns[12] = 1  # ftype: single tilt around the Y axis
                columns[13:15] = tilt_range
            columns[19] = tomo_id
            # 0-based CETS array frame -> 1-based Dynamo positions
            columns[23:26] = [
                coord + TBL_POSITION_OFFSET for coord in origin_3d[coord_index]
            ]
            rows.append(" ".join(self._format_tbl_value(v) for v in columns))

        with open(tbl_path, "w") as dynamo_tbl:
            dynamo_tbl.write("\n".join(rows) + "\n")
        print(f"tbl file successfully written! -> {tbl_path}")
        return tbl_path

    @staticmethod
    def _get_particle_tags(particle_maps: List[ParticleMap]) -> List[int]:
        """The Dynamo tag of each particle, from its file name (particle_<tag>.<ext>).
        Falls back to 1..N if any name does not follow the pattern or tags repeat."""
        tags: List[int] = []
        for particle_map in particle_maps:
            match = re.fullmatch(r"particle_(\d+)", Path(particle_map.path or "").stem)
            if match is None:
                break
            tags.append(int(match.group(1)))
        if len(tags) == len(particle_maps) and len(set(tags)) == len(tags):
            return tags
        return list(range(1, len(particle_maps) + 1))

    @staticmethod
    def _get_voxel_size(transformations: List, particle_ind: int) -> float:
        for transformation in transformations:
            if isinstance(transformation, Scale) and transformation.scale:
                return float(transformation.scale[0])
        raise ValueError(
            f"Particle map {particle_ind} has no array_to_physical scale. "
            f"Please provide pixel_size."
        )

    @staticmethod
    def _format_tbl_value(value: float) -> str:
        # Up to 6 decimals, without trailing zeros (integers are written as such).
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return "0" if text in ("-0", "") else text

    @staticmethod
    def _get_dynamo_euler_angles(matrix: np.ndarray) -> Tuple[float, float, float]:
        """Inverse of ``_get_dynamo_euler_matrix``: (tdrot, tilt, narot) in degrees, with
        tilt in [0, 180], from R = Rz(-tdrot) @ Rx(-tilt) @ Rz(-narot)."""
        tilt = np.arccos(np.clip(matrix[2, 2], -1.0, 1.0))
        if abs(np.sin(tilt)) < 1e-8:
            # Gimbal lock: only tdrot + narot (tilt = 0) or narot - tdrot (tilt = 180)
            # is defined. tdrot is set to 0.
            tdrot = 0.0
            narot = np.arctan2(matrix[1, 0], matrix[0, 0])
            if matrix[2, 2] > 0:
                narot = -narot
        else:
            tdrot = np.arctan2(matrix[0, 2], matrix[1, 2])
            narot = np.arctan2(matrix[2, 0], -matrix[2, 1])
        return (
            float(np.rad2deg(tdrot)),
            float(np.rad2deg(tilt)),
            float(np.rad2deg(narot)),
        )

    @staticmethod
    def _get_dynamo_euler_matrix(line_parts: List) -> np.ndarray:
        # .tbl rows are text, so the angle columns come in as strings; cast to float
        # before any numeric (numpy) operation.
        tdrot = float(line_parts[6])
        tilt = float(line_parts[7])
        narot = float(line_parts[8])
        # Convert the angles to radians
        tdrot = np.deg2rad(tdrot)
        tilt = np.deg2rad(tilt)
        narot = np.deg2rad(narot)
        # Rotations are clockwise
        tdrot *= -1
        tilt *= -1
        narot *= -1
        # Rotation around Z (tdrot)
        Rz1 = np.array(
            [
                [np.cos(tdrot), -np.sin(tdrot), 0],
                [np.sin(tdrot), np.cos(tdrot), 0],
                [0, 0, 1],
            ]
        )
        # Rotation around X (tilt)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(tilt), -np.sin(tilt)],
                [0, np.sin(tilt), np.cos(tilt)],
            ]
        )
        # Rotation around Z (narot)
        Rz2 = np.array(
            [
                [np.cos(narot), -np.sin(narot), 0],
                [np.sin(narot), np.cos(narot), 0],
                [0, 0, 1],
            ]
        )
        # Compose ZXZ
        return Rz1 @ Rx @ Rz2

    def _get_particle_transform(
        self, line_parts: List, physical_cs_name: str
    ) -> Affine:
        euler_matrix = self._get_dynamo_euler_matrix(line_parts)
        euler_matrix_list = euler_matrix.tolist()
        angular_matrix = [
            sublist[:3] for sublist in euler_matrix_list[:3]
        ]  # Take only the angular 3x3 sub-matrix
        # The pose is anchored to the particle physical frame (endomorphism within it).
        return Affine(
            name="Subtomogram orientation",
            affine=angular_matrix,
            input=physical_cs_name,
            output=physical_cs_name,
        )

    @staticmethod
    def _get_particle_translation(
        line_parts: List, physical_cs_name: str, voxel_size: float = 1.0
    ) -> Translation:
        # Shifts converted from pixels to Å (physical frame).
        shift_x = float(line_parts[3]) * voxel_size
        shift_y = float(line_parts[4]) * voxel_size
        shift_z = float(line_parts[5]) * voxel_size
        return Translation(
            translation=[shift_x, shift_y, shift_z],
            name="Dynamo translation from a .tbl file. Shifts in angstroms.",
            input=physical_cs_name,
            output=physical_cs_name,
        )
