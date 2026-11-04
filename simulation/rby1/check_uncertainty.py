"""Check delay timing and isolation of a free payload's rigid-body model."""
import json
import mujoco as mj
import numpy as np
from build_model import OUT, dump
from build_payload import PAY
from control_uncertainty import check_delay


def main():
    q = np.array(json.loads((OUT.parent/'buffer_preflight/initialization.json').read_text())['qpos'])
    values = []
    for scale in [1., 1.4]:
        m = mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml')); ob = m.body('payload').id
        m.body_mass[ob] *= scale; m.body_inertia[ob] *= scale
        d = mj.MjData(m); mj.mj_setConst(m, d); d.qpos[:] = q
        d.qvel[:] = np.linspace(-.02, .02, m.nv); mj.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv)); mj.mj_fullM(m, d, M)
        p = int(m.joint('payload_free').dofadr[0]); robot = np.array([i for i in range(m.nv) if i not in range(p, p+6)])
        values.append((M[np.ix_(robot, robot)], d.qfrc_bias[robot].copy(), M[np.ix_(robot, np.arange(p, p+6))]))
    errors = {'robot_mass_matrix_change': float(np.max(np.abs(values[0][0]-values[1][0]))),
              'robot_bias_force_change': float(np.max(np.abs(values[0][1]-values[1][1]))),
              'robot_free_payload_cross_block': float(np.max(np.abs(values[0][2])))}
    assert max(errors.values()) < 1e-12
    result = {'delay_cases': check_delay(), 'free_body_isolation': errors,
              'scope': 'Unconstrained M and bias blocks only. Physical contact coupling remains active in simulation.'}
    dest = OUT.parent/'stage7'; dest.mkdir(exist_ok=True); dump(dest/'preflight.json', result)
    print(json.dumps(result))


if __name__ == '__main__': main()
