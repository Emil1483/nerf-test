import cv2
import os
from pathlib import Path

import concurrent.futures
import numpy as np
import imageio
from PIL import Image

from absl import logging
from typing import Dict
import numpy as np
from nerfies.camera import Camera
import pycolmap
from pycolmap import Quaternion

save_dir = 'captures'
capture_name = 'capture1'

root_dir = Path(save_dir, capture_name)

rgb_dir = root_dir / 'rgb'
rgb_raw_dir = root_dir / 'rgb-raw'

colmap_dir = root_dir / 'colmap'
colmap_db_path = colmap_dir / 'database.db'
colmap_out_path = colmap_dir / 'sparse'

os.system(f'mkdir {colmap_dir}')

video_path = 'input.mp4'
max_scale = 1.0
fps = 1
target_num_frames = 10

cap = cv2.VideoCapture(video_path)
input_fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if num_frames < target_num_frames:
    raise RuntimeError(
        'The video is too short and has fewer frames than the target.')

if fps == -1:
    fps = int(target_num_frames / num_frames * input_fps)
    print(f"Auto-computed FPS = {fps}")

overwrite = True


if (rgb_dir / '1x').exists() and not overwrite:
    raise RuntimeError(
        f'The RGB frames have already been processed. Check `overwrite` and run again if you really meant to do this.')
else:
    filters = f"mpdecimate,setpts=N/FRAME_RATE/TB,scale=iw*{max_scale}:ih*{max_scale}"
    tmp_rgb_raw_dir = 'rgb-raw'
    out_pattern = str('rgb-raw/%06d.png')
    os.system(f'mkdir "{tmp_rgb_raw_dir}"')
    os.system(
        f'ffmpeg -i "{video_path}" -r {fps} -vf {filters} "{out_pattern}"')
    os.system(f'mkdir "{root_dir}"')
    os.system(f'move "{tmp_rgb_raw_dir}" "{root_dir}"')

# --- Resize images into different scales ---


def save_image(path, image: np.ndarray) -> None:
    print(f'Saving {path}')
    if not path.parent.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('wb') as f:
        image = Image.fromarray(np.asarray(image))
        image.save(f, format=path.suffix.lstrip('.'))


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert the image to a uint8 array."""
    if image.dtype == np.uint8:
        return image
    if not issubclass(image.dtype.type, np.floating):
        raise ValueError(
            f'Input image should be a floating type but is of type {image.dtype!r}')
    return (image * 255).clip(0.0, 255).astype(np.uint8)


def make_divisible(image: np.ndarray, divisor: int) -> np.ndarray:
    """Trim the image if not divisible by the divisor."""
    height, width = image.shape[:2]
    if height % divisor == 0 and width % divisor == 0:
        return image

    new_height = height - height % divisor
    new_width = width - width % divisor

    return image[:new_height, :new_width]


def downsample_image(image: np.ndarray, scale: int) -> np.ndarray:
    """Downsamples the image by an integer factor to prevent artifacts."""
    if scale == 1:
        return image

    height, width = image.shape[:2]
    if height % scale > 0 or width % scale > 0:
        raise ValueError(f'Image shape ({height},{width}) must be divisible by the'
                         f' scale ({scale}).')
    out_height, out_width = height // scale, width // scale
    resized = cv2.resize(image, (out_width, out_height), cv2.INTER_AREA)
    return resized


image_scales = "1,2,4,8"
image_scales = [int(x) for x in image_scales.split(',')]

tmp_rgb_dir = Path('rgb')


for image_path in Path(rgb_raw_dir).glob('*.png'):
    image = make_divisible(imageio.imread(image_path), max(image_scales))
    for scale in image_scales:
        save_image(
            tmp_rgb_dir / f'{scale}x/{image_path.stem}.png',
            image_to_uint8(downsample_image(image, scale)))

os.system(f'move "{tmp_rgb_dir}" "{root_dir}"')

# ---Camera registration with COLMAP ---

share_intrinsics = True
assume_upright_cameras = True

colmap_image_scale = 4
colmap_rgb_dir = rgb_dir / f'{colmap_image_scale}x'

overwrite = False

if overwrite and colmap_db_path.exists():
    colmap_db_path.unlink()

os.system(f'COLMAP.bat feature_extractor ' +
          f'--SiftExtraction.use_gpu 0 ' +
          f'--SiftExtraction.upright {int(assume_upright_cameras)} ' +

          f'--ImageReader.single_camera {int(share_intrinsics)} ' +
          f'--database_path "{colmap_db_path}" ' +
          f'--image_path "{colmap_rgb_dir}"'
          )

# -- Match feature

match_method = 'exhaustive'

if match_method == 'exhaustive':
    os.system('COLMAP.bat exhaustive_matcher ' +
              '--SiftMatching.use_gpu 0 ' +
              f'--database_path "{str(colmap_db_path)}"'
              )
else:
    os.system('COLMAP.bat vocab_tree_matcher ' +
              '--VocabTreeMatching.vocab_tree_path vocab_tree_flickr100K_words32K.bin ' +
              '--SiftMatching.use_gpu 0' +
              f'--database_path "{str(colmap_db_path)}"'
              )

refine_principal_point = True
min_num_matches = 32
filter_max_reproj_error = 2
tri_complete_max_reproj_error = 2

os.system(
    'COLMAP.bat mapper ' +
    f'--Mapper.ba_refine_principal_point {int(refine_principal_point)} ' +
    f'--Mapper.filter_max_reproj_error $filter_max_reproj_error ' +
    f'--Mapper.tri_complete_max_reproj_error $tri_complete_max_reproj_error ' +
    f'--Mapper.min_num_matches $min_num_matches ' +
    f'--database_path "{str(colmap_db_path)}" ' +
    f'--image_path "{str(colmap_rgb_dir)}" ' +
    f'--export_path "{str(colmap_out_path)}"'
)

if not colmap_db_path.exists():
    raise RuntimeError(
        f'The COLMAP DB does not exist, did you run the reconstruction?')
elif not (colmap_dir / 'sparse/0/cameras.bin').exists():
    raise RuntimeError("""
SfM seems to have failed. Try some of the following options:
 - Increase the FPS when flattenting to images. There should be at least 50-ish images.
 - Decrease `min_num_matches`.
 - If you images aren't upright, uncheck `assume_upright_cameras`.
""")
else:
    print("Everything looks good!")


# --- Define Scene Manager


def convert_colmap_camera(colmap_camera, colmap_image):
    """Converts a pycolmap `image` to an SFM camera."""
    camera_rotation = colmap_image.R()
    camera_position = -(colmap_image.t @ camera_rotation)
    new_camera = Camera(
        orientation=camera_rotation,
        position=camera_position,
        focal_length=colmap_camera.fx,
        pixel_aspect_ratio=colmap_camera.fx / colmap_camera.fx,
        principal_point=np.array([colmap_camera.cx, colmap_camera.cy]),
        radial_distortion=np.array([colmap_camera.k1, colmap_camera.k2, 0.0]),
        tangential_distortion=np.array([colmap_camera.p1, colmap_camera.p2]),
        skew=0.0,
        image_size=np.array([colmap_camera.width, colmap_camera.height])
    )
    return new_camera


def filter_outlier_points(points, inner_percentile):
    """Filters outlier points."""
    outer = 1.0 - inner_percentile
    lower = outer / 2.0
    upper = 1.0 - lower
    centers_min = np.quantile(points, lower, axis=0)
    centers_max = np.quantile(points, upper, axis=0)
    result = points.copy()

    too_near = np.any(result < centers_min[None, :], axis=1)
    too_far = np.any(result > centers_max[None, :], axis=1)

    return result[~(too_near | too_far)]


def average_reprojection_errors(points, pixels, cameras):
    """Computes the average reprojection errors of the points."""
    cam_errors = []
    for i, camera in enumerate(cameras):
        cam_error = reprojection_error(points, pixels[:, i], camera)
        cam_errors.append(cam_error)
    cam_error = np.stack(cam_errors)

    return cam_error.mean(axis=1)


def _get_camera_translation(camera):
    """Computes the extrinsic translation of the camera."""
    rot_mat = camera.orientation
    return -camera.position.dot(rot_mat.T)


def _transform_camera(camera, transform_mat):
    """Transforms the camera using the given transformation matrix."""
    # The determinant gives us volumetric scaling factor.
    # Take the cube root to get the linear scaling factor.
    scale = np.cbrt(linalg.det(transform_mat[:, :3]))
    quat_transform = ~Quaternion.FromR(transform_mat[:, :3] / scale)

    translation = _get_camera_translation(camera)
    rot_quat = Quaternion.FromR(camera.orientation)
    rot_quat *= quat_transform
    translation = scale * translation - rot_quat.ToR().dot(transform_mat[:, 3])
    new_transform = np.eye(4)
    new_transform[:3, :3] = rot_quat.ToR()
    new_transform[:3, 3] = translation

    rotation = rot_quat.ToR()
    new_camera = camera.copy()
    new_camera.orientation = rotation
    new_camera.position = -(translation @ rotation)
    return new_camera


def _pycolmap_to_sfm_cameras(manager: pycolmap.SceneManager) -> Dict[int, Camera]:
    """Creates SFM cameras."""
    # Use the original filenames as indices.
    # This mapping necessary since COLMAP uses arbitrary numbers for the
    # image_id.
    image_id_to_colmap_id = {
        image.name.split('.')[0]: image_id
        for image_id, image in manager.images.items()
    }

    sfm_cameras = {}
    for image_id in image_id_to_colmap_id:
        colmap_id = image_id_to_colmap_id[image_id]
        image = manager.images[colmap_id]
        camera = manager.cameras[image.camera_id]
        sfm_cameras[image_id] = convert_colmap_camera(camera, image)

    return sfm_cameras


class SceneManager:
    """A thin wrapper around pycolmap."""

    @classmethod
    def from_pycolmap(cls, colmap_path, image_path, min_track_length=10):
        """Create a scene manager using pycolmap."""
        manager = pycolmap.SceneManager(str(colmap_path))
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()
        manager.filter_points3D(min_track_len=min_track_length)
        sfm_cameras = _pycolmap_to_sfm_cameras(manager)
        return cls(sfm_cameras, manager.get_filtered_points3D(), image_path)

    def __init__(self, cameras, points, image_path):
        self.image_path = Path(image_path)
        self.camera_dict = cameras
        self.points = points

        logging.info('Created scene manager with %d cameras',
                     len(self.camera_dict))

    def __len__(self):
        return len(self.camera_dict)

    @property
    def image_ids(self):
        return sorted(self.camera_dict.keys())

    @property
    def camera_list(self):
        return [self.camera_dict[i] for i in self.image_ids]

    @property
    def camera_positions(self):
        """Returns an array of camera positions."""
        return np.stack([camera.position for camera in self.camera_list])

    def load_image(self, image_id):
        """Loads the image with the specified image_id."""
        path = self.image_path / f'{image_id}.png'
        with path.open('rb') as f:
            return imageio.imread(f)

    def triangulate_pixels(self, pixels):
        """Triangulates the pixels across all cameras in the scene.

        Args:
          pixels: the pixels to triangulate. There must be the same number of pixels
            as cameras in the scene.

        Returns:
          The 3D points triangulated from the pixels.
        """
        if pixels.shape != (len(self), 2):
            raise ValueError(
                f'The number of pixels ({len(pixels)}) must be equal to the number '
                f'of cameras ({len(self)}).')

        return triangulate_pixels(pixels, self.camera_list)

    def change_basis(self, axes, center):
        """Change the basis of the scene.

        Args:
          axes: the axes of the new coordinate frame.
          center: the center of the new coordinate frame.

        Returns:
          A new SceneManager with transformed points and cameras.
        """
        transform_mat = np.zeros((3, 4))
        transform_mat[:3, :3] = axes.T
        transform_mat[:, 3] = -(center @ axes)
        return self.transform(transform_mat)

    def transform(self, transform_mat):
        """Transform the scene using a transformation matrix.

        Args:
          transform_mat: a 3x4 transformation matrix representation a
            transformation.

        Returns:
          A new SceneManager with transformed points and cameras.
        """
        if transform_mat.shape != (3, 4):
            raise ValueError(
                'transform_mat should be a 3x4 transformation matrix.')

        points = None
        if self.points is not None:
            points = self.points.copy()
            points = points @ transform_mat[:, :3].T + transform_mat[:, 3]

        new_cameras = {}
        for image_id, camera in self.camera_dict.items():
            new_cameras[image_id] = _transform_camera(camera, transform_mat)

        return SceneManager(new_cameras, points, self.image_path)

    def filter_images(self, image_ids):
        num_filtered = 0
        for image_id in image_ids:
            if self.camera_dict.pop(image_id, None) is not None:
                num_filtered += 1

        return num_filtered
