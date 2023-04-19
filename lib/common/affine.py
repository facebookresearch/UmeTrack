import numpy as np
from scipy.spatial.transform import Rotation


def transform3(m, v):
    return transform_vec3(m, v) + m[..., :3, 3]


def transform_vec3(m, v):
    if m.ndim == 2:
        return (v.reshape(-1, 3) @ m[:3, :3].T).reshape(v.shape)
    else:
        return (m[..., :3, :3] @ v[..., None]).squeeze(-1)


def normalized(v: np.ndarray, axis: int = -1, eps: float = 5.43e-20) -> np.ndarray:
    d = np.maximum(eps, (v * v).sum(axis=axis, keepdims=True) ** 0.5)
    return v / d


def skew_matrix(v: np.ndarray) -> np.ndarray:
    res = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=v.dtype
    )
    return res


def from_two_vectors(a_orig: np.ndarray, b_orig: np.ndarray) -> np.ndarray:
    a = normalized(a_orig)
    b = normalized(b_orig)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    v_mat = skew_matrix(v)

    rot = np.eye(3, 3) + v_mat + np.matmul(v_mat, v_mat) * (1 - c) / (max(s * s, 1e-15))

    return rot


def make_look_at_matrix(
    orig_world_to_eye: np.ndarray,
    center: np.ndarray,
    camera_angle: float = 0,
) -> np.ndarray:
    center_local = transform3(orig_world_to_eye, center)
    z_dir_local = center_local / np.linalg.norm(center_local)
    delta_r_local = from_two_vectors(
        np.array([0, 0, 1], dtype=center.dtype), z_dir_local
    )
    orig_eye_to_world = np.linalg.inv(orig_world_to_eye)

    new_eye_to_world = orig_eye_to_world.copy()
    new_eye_to_world[0:3, 0:3] = orig_eye_to_world[0:3, 0:3] @ delta_r_local

    # Locally rotate the z axis to align with the camera angle
    z_local_rot = Rotation.from_euler("z", camera_angle, degrees=True).as_matrix()
    new_eye_to_world[0:3, 0:3] = new_eye_to_world[0:3, 0:3] @ z_local_rot

    return np.linalg.inv(new_eye_to_world)
