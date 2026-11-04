"""Small independent OTG, transition and geometry checks before physical runs."""
import hashlib,json
import numpy as np
import mujoco as mj
from pathlib import Path
from state_stop_reference import make_otg,LegacyStateStop as StateStop,carry_geoms,upper_front
from stop_profiles import parameters,stop
from build_model import ROOT,OUT,dump
from build_payload import PAY


def main():
    dest=OUT.parent/'stage14';dest.mkdir(exist_ok=True);cases=[]
    for v in [.2,.4,.5]:
        p=parameters(v,.78,5.);tr=make_otg(v,0.,minimum_duration=0.)
        errors=[]
        for t in np.linspace(0,p['duration_s'],301):
            a=np.array(tr.at_time(t))[:,0];b=np.array(stop(t,p));errors.append(float(max(abs(a-b))))
        assert max(errors)<1e-10 and abs(tr.duration-p['duration_s'])<1e-12
        cases.append({'velocity':v,'zero_acceleration_equivalence_error':max(errors),'duration_s':tr.duration})
    tr=make_otg(.2,.6,minimum_duration=.6);z=np.array([np.array(tr.at_time(t))[:,0] for t in np.linspace(0,tr.duration,2001)])
    assert abs(z[0,2]-.6)<1e-12 and np.max(np.abs(z[-1,1:]))<1e-10 and max(abs(z[:,2]))<=.78+1e-10
    rate=np.diff(z[:,2])/(tr.duration/2000);assert max(abs(rate))<=5.+1e-8
    m=mj.MjModel.from_xml_path(str(PAY/'payload_scene.xml'));d=mj.MjData(m)
    d.qpos[:]=json.loads((OUT.parent/'buffer_preflight/initialization.json').read_text())['qpos'];mj.mj_forward(m,d)
    ids=carry_geoms(m);mesh_bound_error=[]
    for g in ids:
        if m.geom_type[g]!=mj.mjtGeom.mjGEOM_MESH:continue
        mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);end=start+int(m.mesh_vertnum[mesh])
        vertices=m.mesh_vert[start:end]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
        projected=upper_front(m,d,np.array([g]));mesh_bound_error.append(float(np.max(vertices[:,0])-projected))
    assert max(mesh_bound_error)<1e-6
    wheel=ROOT/'work/robot-model-audit/packages/ruckig-0.19.4-cp312-cp312-win_amd64.whl'
    h=hashlib.sha256(wheel.read_bytes()).hexdigest();assert h=='9ca62445749dbcdb1533eb44ef8d434bdd43c33f7e274c9117698221c9c4d620'
    result={'ruckig_scurve_equivalence':cases,'nonzero_acceleration':{'initial_a':float(z[0,2]),'duration':tr.duration,'max_a':float(max(abs(z[:,2]))),'max_jerk':float(max(abs(rate)))},
            'upper_geometry_count':len(ids),'mesh_geometry_checks':len(mesh_bound_error),'largest_mesh_vertex_outside_aabb_m':max(mesh_bound_error),
            'wheel_sha256':h,'ruckig_version':'0.19.4','source_url':'https://pypi.org/project/ruckig/0.19.4/'}
    dump(dest/'independent_checks.json',result);print(json.dumps(result))


if __name__=='__main__':main()
