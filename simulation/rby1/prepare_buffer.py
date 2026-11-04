"""Reproduce the shared close-to-body initial grasp for stage 5."""
import json
import mujoco as mj
from build_model import OUT,dump
from build_payload import PAY,solve_ik
from buffer_reference import arm_path,check_reference


def main():
    check_reference()
    m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'))
    initial=json.loads((PAY/'initialization.json').read_text());d=mj.MjData(m);d.qpos[:]=initial['qpos'];mj.mj_forward(m,d)
    refs={s:(d.site_xpos[m.site('ee_'+s+'_site').id].copy(),d.site_xmat[m.site('ee_'+s+'_site').id].reshape(3,3).copy()) for s in ['left','right']}
    d.qpos[m.joint('payload_free').qposadr[0]]-=.1
    ik=[solve_ik(m,d,s,p+[-.1,0,0],R) for s,(p,R) in refs.items()]
    assert all(r['position_error_m']<1e-5 and r['angle_error_rad']<1e-4 for r in ik)
    initial['qpos']=d.qpos.tolist();initial['ik']=ik
    initial['stage5_initialization_note']='Payload and both hands moved 0.10 m toward robot, to provide arm reach for a shared initial condition. Physical state initialization only.'
    initial['buffer_path_audit']=arm_path(m,initial['qpos'])['audit']
    dest=OUT.parent/'buffer_preflight';dest.mkdir(exist_ok=True);dump(dest/'initialization.json',initial)
    print(json.dumps(initial['buffer_path_audit']))


if __name__=='__main__':main()
