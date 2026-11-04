"""World-frame reference contract; independent of optimizer and physical state.

Base positions refer to the wheel axle midpoint; payload positions to its COM.
All angular velocities/accelerations are world expressed. SI units throughout.
Callbacks describe a complete offline plan and hold their final state after T.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco as mj
import numpy as np


@dataclass
class WholeBodyStopReference:
    T: float                         # base stop time; stop-priority feedback gate
    T_all: float                     # end of all planned motion
    base_at: Callable                # p, yaw, world v, a, omega, alpha
    payload_at: Callable             # COM p, v, a, R, world omega, alpha
    joint_at: Callable | None        # upper-body q, v, a; None means old hold
    qadr: np.ndarray
    vadr: np.ndarray
    config: dict
    source_paths: tuple

    def at(self, t):
        return self.base_at(t)

    def payload(self, t):
        return self.payload_at(t)

    def joint(self, t):
        return None if self.joint_at is None else self.joint_at(t)

    @classmethod
    def from_legacy(cls, model, q0, base, upper=None):
        """Adapt the old analytic references without re-sampling/interpolation."""
        from qualify_planar_stop import rz
        d = mj.MjData(model); d.qpos[:] = q0; mj.mj_kinematics(model, d)
        obj = model.body('payload').id
        p0, yaw0, *_ = base.at(0.)
        offset = rz(-yaw0) @ (d.xipos[obj]-p0)
        rotation0 = d.xmat[obj].reshape(3, 3).copy()

        def payload(t):
            planned = base.at(t)
            p, yaw, v, a, w, al = planned
            r = rz(yaw) @ offset
            position = p+r
            velocity = v+np.cross(w, r)
            acceleration = a+np.cross(al, r)+np.cross(w, np.cross(w, r))
            if upper is not None:
                position, velocity, acceleration = upper.com(t, planned, offset)
            return position, velocity, acceleration, rz(yaw-yaw0) @ rotation0, w, al

        paths = [Path(__file__), Path(__file__).with_name('qualify_planar_stop.py')]
        if upper is not None:
            paths += [upper.source_path, Path(__file__).with_name('motion_distribution.py'),
                      Path(__file__).with_name('buffer_reference.py')]
        return cls(base.T, base.T if upper is None else max(base.T, upper.T),
                   base.at, payload, None if upper is None else upper.joint,
                   np.array([], dtype=int) if upper is None else upper.path['qadr'],
                   np.array([], dtype=int) if upper is None else upper.path['vadr'],
                   {'kind': 'analytic_legacy_adapter', 'units': 'SI',
                    'base_frame': 'world axle midpoint', 'payload_frame': 'world COM',
                    'angular_frame': 'world', 'force_distribution': 'unchanged equal split',
                    'upper': None if upper is None else upper.config}, tuple(paths))


def validate(reference, model, q0):
    """Numerical contract check only, not geometric or dynamic admission."""
    assert 0 < reference.T <= reference.T_all
    assert reference.qadr.shape == reference.vadr.shape
    assert len(np.unique(reference.qadr)) == len(reference.qadr)
    assert np.all((reference.qadr >= 0) & (reference.qadr < model.nq))
    assert np.all((reference.vadr >= 0) & (reference.vadr < model.nv))
    maximum_rotation_error = terminal_speed = terminal_acceleration = 0.
    for t in np.unique(np.r_[np.linspace(0., reference.T_all, 31), reference.T, reference.T_all+1.]):
        b = reference.at(float(t)); o = reference.payload(float(t)); u = reference.joint(float(t))
        assert len(b) == 6 and np.ndim(b[1]) == 0
        assert all(np.shape(b[i]) == (3,) for i in [0, 2, 3, 4, 5])
        assert len(o) == 6 and np.shape(o[3]) == (3, 3)
        assert all(np.shape(o[i]) == (3,) for i in [0, 1, 2, 4, 5])
        assert all(np.all(np.isfinite(x)) for x in (*b, *o))
        maximum_rotation_error = max(maximum_rotation_error, float(np.max(abs(o[3].T @ o[3]-np.eye(3)))))
        assert np.linalg.det(o[3]) > 0
        if u is not None:
            assert len(u) == 3 and all(np.shape(x) == reference.qadr.shape and np.all(np.isfinite(x)) for x in u)
        if t >= reference.T_all:
            terminal_speed = max(terminal_speed, *(float(np.linalg.norm(x)) for x in [b[2], b[4], o[1], o[4]]),
                                 0. if u is None else float(np.linalg.norm(u[1])))
            terminal_acceleration = max(terminal_acceleration, *(float(np.linalg.norm(x)) for x in [b[3], b[5], o[2], o[5]]),
                                        0. if u is None else float(np.linalg.norm(u[2])))
    u0 = reference.joint(0.)
    initial_joint_error = 0. if u0 is None else float(np.max(abs(u0[0]-q0[reference.qadr])))
    assert maximum_rotation_error < 1e-10 and max(terminal_speed, terminal_acceleration, initial_joint_error) < 1e-9
    return {'rotation_orthogonality_error': maximum_rotation_error, 'initial_joint_error_rad': initial_joint_error,
            'terminal_reference_speed_norm': terminal_speed, 'terminal_reference_acceleration_norm': terminal_acceleration,
            'scope': 'shape, finite values, rotation, initial joints and terminal rest; not physical tracking or collision proof'}
