"""Geometric visibility of objects from a fixed camera, by ray casting.

Used instead of rendering so that "the camera cannot see it" is cheap to
evaluate at every look and agrees with what the rendered image shows.
"""

import mujoco
import numpy as np

_ROBOT_PREFIXES = ("robot0", "gripper0", "mount0")
_EPS = 1e-4


def _raw(env):
    sim = env.env.sim
    return sim.model._model, sim.data._data


def _rendered_groups(env):
    """Geom groups the offscreen renderer draws (robosuite hides collision geoms)."""
    return np.array(env.env.sim._render_context_offscreen.vopt.geomgroup, dtype=np.uint8)


def _in_view(model, data, cam_id, points):
    """Which world points project inside the camera image (square aspect)."""
    rel = (points - data.cam_xpos[cam_id]) @ data.cam_xmat[cam_id].reshape(3, 3)
    depth = -rel[:, 2]  # MuJoCo cameras look along their -z axis
    half = np.tan(np.radians(model.cam_fovy[cam_id]) / 2) * depth
    return (depth > 0) & (np.abs(rel[:, 0]) <= half) & (np.abs(rel[:, 1]) <= half)


def _surface_points(model, data, root, origin, n_points=300):
    """Points on the object's camera-facing surface, spread evenly over the
    area it presents to the camera.

    Sampled from the triangles of its visual meshes, weighted by projected
    area. Normals are oriented away from the mesh centroid, which is right for
    the roughly convex objects used here. Falls back to geom centres for
    objects without meshes.
    """
    triangles, weights = [], []
    for g in range(model.ngeom):
        if model.body_rootid[model.geom_bodyid[g]] != root:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[g]
        vadr, vnum = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        fadr, fnum = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        verts = data.geom_xpos[g] + model.mesh_vert[vadr : vadr + vnum] @ data.geom_xmat[g].reshape(3, 3).T
        tri = verts[model.mesh_face[fadr : fadr + fnum]]  # (faces, 3, 3)
        centre = tri.mean(axis=1)
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # length = 2 * area
        outward = centre - verts.mean(axis=0)
        normal *= np.sign(np.einsum("ij,ij->i", normal, outward))[:, None]
        to_cam = origin - centre
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
        projected = np.einsum("ij,ij->i", normal, to_cam)
        facing = projected > 0
        triangles.append(tri[facing])
        weights.append(projected[facing])
    if not triangles:
        return np.array(
            [data.geom_xpos[g] for g in range(model.ngeom)
             if model.body_rootid[model.geom_bodyid[g]] == root]
        )
    triangles, weights = np.concatenate(triangles), np.concatenate(weights)

    # Stratified pick of triangles, then a fixed pseudo-random point inside each;
    # deterministic so that visibility is a pure function of the scene.
    rng = np.random.default_rng(0)
    cdf = np.cumsum(weights) / weights.sum()
    chosen = np.searchsorted(cdf, (np.arange(n_points) + 0.5) / n_points)
    a, b = rng.random(n_points), rng.random(n_points)
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    tri = triangles[chosen]
    return tri[:, 0] + a[:, None] * (tri[:, 1] - tri[:, 0]) + b[:, None] * (tri[:, 2] - tri[:, 0])


def _occluded(model, data, origin, direction, max_dist, own_body, robot_roots, groups):
    """Whether something other than the robot or the object itself blocks the ray.

    Only geoms in the rendered `groups` count, so that what blocks a ray is
    exactly what would cover the object in the image.
    """
    geomid = np.zeros(1, dtype=np.int32)
    travelled = 0.0
    pnt = origin
    for _ in range(50):  # each iteration skips one robot geom
        dist = mujoco.mj_ray(model, data, pnt, direction, groups, 1, own_body, geomid)
        if geomid[0] < 0 or travelled + dist >= max_dist:
            return False
        if model.body_rootid[model.geom_bodyid[geomid[0]]] not in robot_roots:
            return True
        travelled += dist + _EPS
        pnt = pnt + (dist + _EPS) * direction
    return False


def visible_fraction(env, name: str, camera: str = "agentview") -> float:
    """Fraction of the object's camera-facing surface the camera actually sees.

    Points outside the image count as not visible. The robot is treated as
    transparent so that the arm passing in front does not hide objects.
    """
    model, data = _raw(env)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    body = env.env.obj_body_id[name]
    origin = data.cam_xpos[cam_id].copy()
    robot_roots = {
        r for r in set(model.body_rootid) if model.body(r).name.startswith(_ROBOT_PREFIXES)
    }
    groups = _rendered_groups(env)

    points = _surface_points(model, data, model.body_rootid[body], origin)
    total = len(points)
    points = points[_in_view(model, data, cam_id, points)]
    to_points = points - origin
    dists = np.linalg.norm(to_points, axis=1)
    directions = to_points / dists[:, None]
    # mj_multiRay would be faster but misses geoms here (checked against mj_ray).
    visible = sum(
        not _occluded(model, data, origin, directions[i], dists[i], body, robot_roots, groups)
        for i in range(len(points))
    )
    return visible / total
