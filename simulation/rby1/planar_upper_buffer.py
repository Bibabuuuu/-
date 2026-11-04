"""Optional local torso/arm buffer reference, not an online whole-body optimizer."""
from pathlib import Path
import mujoco as mj
import numpy as np

from buffer_reference import evaluate
from motion_distribution import motion_path
from qualify_planar_stop import PlanarStop, rz
from state_stop_reference import make_otg
from stop_scene_chain import frame, reference_pose


class PlanarUpperBuffer:
    def __init__(self, m, q0, v0, base_A=1., object_A=.5, path=None):
        _, self.axle0, self.yaw0, speed = frame(m, q0, v0)
        self.base = make_otg(speed, 0., base_A, 5.)
        self.object = make_otg(speed, 0., object_A, 5.)
        self.T = max(self.base.duration, self.object.duration)
        extent = self.relative(self.T)[0]
        assert extent > 0 and self.object.duration > self.base.duration
        if path is None:
            # Normalize yaw for the old world-x IK. Joint angles then transfer
            # back to every planar heading; root roll/pitch are preserved.
            initial = q0.copy()
            dq = np.array([np.cos(-self.yaw0/2), 0., 0., np.sin(-self.yaw0/2)])
            for k in [0, int(m.joint("payload_free").qposadr[0])]:
                initial[k:k+3] = self.axle0+rz(-self.yaw0) @ (q0[k:k+3]-self.axle0)
                mj.mju_mulQuat(initial[k+3:k+7], dq, q0[k+3:k+7])
            path = motion_path(m, initial, extent=extent, backward_extent=0., include_torso=True)
        self.path = path
        assert abs(path["extent"]-extent) < 1e-10
        self.source_path = Path(__file__)
        self.config = {"base_A_mps2": base_A, "object_scalar_A_mps2": object_A,
            "J_mps3": 5., "base_stop_s": self.base.duration, "object_stop_s": self.object.duration,
            "forward_stroke_m": extent, "joint_path_audit": path["audit"],
            "policy": "relative scalar stopping distance along current planned base heading; fixed payload orientation in yaw frame; offline shared-object IK; no measured-base pose correction"}

    @staticmethod
    def scalar(otg, t):
        p, v, a = otg.at_time(np.clip(t, 0., otg.duration))
        return np.array([p[0], v[0] if t < otg.duration else 0., a[0] if t < otg.duration else 0.])

    def relative(self, t):
        return self.scalar(self.object, t)-self.scalar(self.base, t)

    def joint(self, t):
        return evaluate(self.path, *self.relative(t))

    def com(self, t, planned, com_offset):
        p, yaw, v, a, w, al = planned
        x, xd, xdd = self.relative(t)
        R = rz(yaw); r = R @ (com_offset+np.array([x, 0., 0.]))
        relative_v = R @ np.array([xd, 0., 0.])
        position = p+r
        velocity = v+np.cross(w, r)+relative_v
        acceleration = a+np.cross(al, r)+np.cross(w, np.cross(w, r))+2*np.cross(w, relative_v)+R @ np.array([xdd, 0., 0.])
        return position, velocity, acceleration

    def pose(self, m, q0, ref, t):
        q = reference_pose(m, q0, self.axle0, self.yaw0, ref, t)
        q[self.path["qadr"]] = self.joint(t)[0]
        k = int(m.joint("payload_free").qposadr[0])
        q[k:k+3] += rz(ref.at(t)[1])[:, 0]*self.relative(t)[0]
        return q
