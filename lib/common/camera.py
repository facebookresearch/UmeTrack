import abc
import json
import math
from typing import NamedTuple, Sequence, Tuple, Type

import numpy as np
from typing_extensions import Protocol, runtime_checkable

from . import affine


class CameraProjection(Protocol):
    """
    Defines a projection from a 3D `xyz` direction or point to 2D.
    """

    @classmethod
    @abc.abstractmethod
    def project(cls, v):
        """
        Project a 3d vector in eye space down to 2d.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def unproject(cls, p):
        """
        Unproject a 2d point to a unit-length vector in eye space.

        `project(unproject(p)) == p`
        `unproject(project(v)) == v / |v|`
        """
        ...


@runtime_checkable
class DistortionModel(Protocol):
    @abc.abstractmethod
    def evaluate(self: Sequence[float], p: np.ndarray) -> np.ndarray:
        """
        Arguments
        ---------
        p: ndarray[..., 2]
            Array of 2D points, of arbitrary batch shape.

        Returns
        -------
        q: ndarray[..., 2]
            Distorted points with same shape as input
        """
        ...


class PerspectiveProjection(CameraProjection):
    @staticmethod
    def project(v):
        # map to [x/z, y/z]
        assert v.shape[-1] == 3
        return v[..., :2] / v[..., 2, None]

    @staticmethod
    def unproject(p):
        # map to [u,v,1] and renormalize
        assert p.shape[-1] == 2
        x, y = np.moveaxis(p, -1, 0)
        v = np.stack((x, y, np.ones(shape=x.shape, dtype=x.dtype)), axis=-1)
        v = affine.normalized(v, axis=-1)
        return v


class ArctanProjection(CameraProjection):
    @staticmethod
    def project(p, eps: float = 2.0**-128):
        assert p.shape[-1] == 3
        x, y, z = np.moveaxis(p, -1, 0)
        r = np.sqrt(x * x + y * y)
        s = np.arctan2(r, z) / np.maximum(r, eps)
        return np.stack((x * s, y * s), axis=-1)

    @staticmethod
    def unproject(uv):
        assert uv.shape[-1] == 2
        u, v = np.moveaxis(uv, -1, 0)
        r = np.sqrt(u * u + v * v)
        c = np.cos(r)
        s = np.sinc(r / np.pi)
        return np.stack([u * s, v * s, c], axis=-1)


class NoDistortion(NamedTuple):
    """
    A trivial distortion model that does not distort the incoming rays.
    """

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        return p


class Fisheye62CameraModel(NamedTuple):
    """
    Fisheye62CameraModel model, with 6 radial and 2 tangential coeffs.
    """

    k1: float
    k2: float
    k3: float
    k4: float
    p1: float
    p2: float
    k5: float
    k6: float

    def evaluate(self: Sequence[float], p: np.ndarray) -> np.ndarray:
        k1, k2, k3, k4, p1, p2, k5, k6 = self
        # radial component
        r2 = (p * p).sum(axis=-1, keepdims=True)
        r2 = np.clip(r2, -np.pi**2, np.pi**2)
        r4 = r2 * r2
        r6 = r2 * r4
        r8 = r4 * r4
        r10 = r4 * r6
        r12 = r6 * r6
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12
        uv = p * radial

        # tangential component
        x, y = uv[..., 0], uv[..., 1]
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        x += 2 * p2 * xy + p1 * (r2 + 2 * x2)
        y += 2 * p1 * xy + p2 * (r2 + 2 * y2)
        return np.stack((x, y), axis=-1)


# ---------------------------------------------------------------------
# API Conventions and naming
#
# Points have the xyz or uv components in the last axis, and may have
# arbitrary batch shapes. ([...,2] for 2d and [...,3] for 3d).
#
# v
#    3D xyz position in eye space, usually unit-length.
# p
#    projected uv coordinates: `p = project(v)`
# q
#    distorted uv coordinates: `q = distort(p)`
# w
#    window coordinates: `q = q * f + [cx, cy]`
#
# A trailing underscore (e.g. `p_`, `q_`) should be read as "hat", and
# generally indicates an approximation to another value.
# ---------------------------------------------------------------------


class CameraModel(CameraProjection, abc.ABC):
    """
    Parameters
    ----------
    width, height : int
        Size of the sensor window

    f : float or tuple(float, float)
        Focal length

    c : tuple(float, float)
        Optical center in window coordinates

    distort_coeffs
        Forward distortion coefficients (eye -> window).

        If this is an instance of DistortionModel, it will be used as-is
        (even if it's a different polynomial than this camera model
        would normally use.) If it's a simple tuple or array, it will
        used as coefficients for `self.distortion_model`.

    camera_to_world_xf : np.ndarray
        Camera's position and orientation in world space, represented as
        a 3x4 or 4x4 matrix.

        The matrix be a rigid transform (only rotation and translation).

        You can change a camera's camera_to_world_xf after construction by
        assigning to or modifying this matrix.

    Attributes
    ----------
    Most attributes are the same as constructor parameters.

    distortion_model
        Class attribute giving the distortion model for new instances.

    """

    width: int
    height: int

    f: Tuple[float, float]
    c: Tuple[float, float]

    camera_to_world_xf: np.ndarray

    distortion_model: Type[DistortionModel]
    distort: DistortionModel

    def __init__(
        self,
        width,
        height,
        f,
        c,
        distort_coeffs,
        camera_to_world_xf=None,
    ):  # pylint: disable=super-init-not-called (see issue 4790 on pylint github)
        self.width = width
        self.height = height

        # f can be either a scalar or (fx,fy) pair. We only fit scalars,
        # but may load (fx, fy) from a stored file.
        self.f = tuple(np.broadcast_to(f, 2))
        self.c = tuple(c)

        if camera_to_world_xf is None:
            self.camera_to_world_xf = np.eye(4)
        else:
            self.camera_to_world_xf = camera_to_world_xf

        if isinstance(distort_coeffs, DistortionModel):
            self.distort = distort_coeffs
        else:
            self.distort = self.distortion_model(*distort_coeffs)

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.width}x{self.height}, f={self.f} c={self.c}"
        )

    def copy(self, camera_to_world_xf=None):
        """Return a copy of this camera

        Arguments
        ---------
        camera_to_world_xf : 4x4 np.ndarray
            Optional new camera_to_world_xf for the new camera model.
            Default is to copy this camera's camera_to_world_xf.
        """
        return self.crop(0, 0, self.width, self.height, camera_to_world_xf=camera_to_world_xf)

    def world_to_eye(self, v):
        """
        Apply camera camera_to_world_xf to points `v` to get eye coords
        """
        return affine.transform_vec3(self.camera_to_world_xf.T, v - self.camera_to_world_xf[:3, 3])

    def eye_to_world(self, v):
        """
        Apply inverse camera camera_to_world_xf to eye points `v` to get world coords
        """
        return affine.transform3(self.camera_to_world_xf, v)

    def eye_to_window(self, v):
        """Project eye coordinates to 2d window coordinates"""
        p = self.project(v)
        q = self.distort.evaluate(p)
        return q * self.f + self.c

    def window_to_eye(self, w):
        """Unproject 2d window coordinates to unit-length 3D eye coordinates"""
        q = (np.asarray(w) - self.c) / self.f
        assert isinstance(
            self.distort, NoDistortion
        ), "Only unprojection for NoDistortion camera is supported"
        return self.unproject(q)

    def crop(
        self,
        src_x,
        src_y,
        target_width,
        target_height,
        scale=1,
        camera_to_world_xf=None,
    ):
        """
        Return intrinsics for a crop of the sensor image.

        No scaling is applied; this just returns the model for a sub-
        array of image data. (Or for a larger array, if (x,y)<=0 and
        (width, height) > (self.width, self.height).

        To do both cropping and scaling, use :meth:`subrect`

        Parameters
        ----------
        x, y, width, height
            Location and size in this camera's window coordinates
        """
        return type(self)(
            target_width,
            target_height,
            np.asarray(self.f) * scale,
            (np.array(self.c) - (src_x, src_y) + 0.5) * scale - 0.5,
            self.distort,
            self.camera_to_world_xf if camera_to_world_xf is None else camera_to_world_xf,
        )


# Camera models
# =============


class PinholePlaneCameraModel(PerspectiveProjection, CameraModel):
    distortion_model = NoDistortion

    def uv_to_window_matrix(self):
        """Return the 3x3 intrinsics matrix"""
        return np.array(
            [[self.f[0], 0, self.c[0]], [0, self.f[1], self.c[1]], [0, 0, 1]]
        )


class Fisheye62CameraModel(ArctanProjection, CameraModel):
    distortion_model = Fisheye62CameraModel


def read_camera_from_json(js):
    if isinstance(js, str):
        js = json.loads(js)
    js = js.get("Camera", js)

    width = js["ImageSizeX"]
    height = js["ImageSizeY"]
    model = js["DistortionModel"]
    fx = js["fx"]
    fy = js["fy"]
    cx = js["cx"]
    cy = js["cy"]

    if model == "PinholePlane":
        cls = PinholePlaneCameraModel
    elif model == "FishEye62":
        cls = Fisheye62CameraModel

    distort_params = cls.distortion_model._fields
    coeffs = [js[name] for name in distort_params]

    return cls(width, height, (fx, fy), (cx, cy), coeffs)
