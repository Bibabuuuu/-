"""Finish existing stage records; preserve all prior experimental evidence."""
import hashlib,importlib.util,json
from pathlib import Path
import mujoco as mj
import numpy as np
from build_model import OUT,HERE,ROOT,dump
from load_sharing import allocate


def main():
    dest=OUT.parent/'stage11';summary=json.loads((dest/'summary.json').read_text());r=summary['runs']
    expected=['双臂分担与躯干协调_文献和仿真报告.md','summary.json','runs.csv','qp_validation.json','posture_screen.json',
              'literature_sources.json','search_metadata_raw.json','figure_qa.json','offset_static_baselines.pdf','posture_torque_tradeoff.pdf']
    for name in expected:assert (dest/name).is_file(),name
    for name in ['offset_static_baselines','posture_torque_tradeoff']:
        qa=json.loads((dest/(name+'.collision.json')).read_text());assert qa['verdict']=='PASS',qa
    for mode in ['fixed','buffer']:
        latest=r[f's11_final_equal_{mode}']
        assert latest['runner_sha256']==hashlib.sha256((HERE/'run_buffer.py').read_bytes()).hexdigest()
    # The geometric baseline added a selector after the first QP experiments.
    # Check the final QP implementation against that archived implementation on recorded states.
    oldrun=r['s11_offset_qp_buffer'];folder=Path(oldrun['folder'])
    spec=importlib.util.spec_from_file_location('stage11_archived_allocator',folder/'load_sharing_snapshot.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    m=mj.MjModel.from_xml_path(str(folder/'scene.xml'));d=mj.MjData(m);z=np.load(folder/'joint_trace.npz')
    acts=[a for a in range(m.nu) if not any(s in m.joint(int(m.actuator_trnid[a,0])).name for s in ['finger','wheel'])]
    joints=[int(m.actuator_trnid[a,0]) for a in acts];va=[int(m.jnt_dofadr[j]) for j in joints]
    caps=np.array([max(abs(m.jnt_actfrcrange[j])) for j in joints]);jp=np.zeros((3,m.nv));jr=jp.copy();compat=[]
    for i in np.linspace(0,len(z['time'])-1,12,dtype=int):
        d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];mj.mj_forward(m,d)
        sites={s:d.site_xpos[m.site(f'ee_{s}_site').id].copy() for s in ['left','right']};blocks=[]
        for side in sites:
            mj.mj_jacSite(m,d,jp,jr,m.site(f'ee_{side}_site').id);blocks.append(np.c_[jp[:,va].T,jr[:,va].T])
        args=(d.xipos[m.body('payload').id],sites,{s:d.geom_xpos[m.geom('payload_'+s+'_handle').id] for s in sites},
              np.array([-.78,0.,9.81]),np.array([.01,.003,.002]),np.hstack(blocks),d.qfrc_bias[va],caps)
        a=old.allocate(*args);b=allocate(*args)
        assert all(np.array_equal(a[k][side],b[k][side]) for k in [0,1] for side in sites)
        compat.append({'recorded_time_s':float(z['time'][i]),'same_wrenches':True})
    dump(dest/'final_qa.json',{'stage_run_count':21,'snapshot_hashes_verified':summary['all_snapshot_hashes_verified'],
          'final_default_regression':summary['nominal_exact_regression'],'final_qp_vs_archived_state_checks':compat,
          'report_has_no_control_characters':not any(ord(c)<32 and c not in '\n\r' for c in (dest/expected[0]).read_text(encoding='utf-8')),
          'delivery_hashes':{n:hashlib.sha256((dest/n).read_bytes()).hexdigest() for n in expected}})
    authority=ROOT/'outputs/robot-topic-study/双课题范围更新.json';p=json.loads(authority.read_text(encoding='utf-8'))
    p['active_focus']['next_phase_status']='第十一小节已完成：文献驱动的双臂分担/躯干作用验证；下一步能力瓶颈与强基线边界待讨论，尚未实施主动躯干优化。'
    a=p['direction_a'];a['status']='当前唯一推进方向：轮式双臂受控制动持物稳定；名义仿真预研至第十一小节。重载载荷族、主动腰臂协调和论文创新仍待验证。'
    a['implementation_status']='第5–11小节有限名义验证已实现，包括载荷补偿、缓冲参考、静态校准、延迟余量、双臂分担和初始躯干姿态对照。几何强基线接近QP；尚无重载、主动躯干轨迹、真机或创新证明。'
    a['physical_meaning_stage']['next_section']['status']='已完成'
    a['literature_load_sharing_stage'].update({'date':'2026-10-31','status':'已完成','scheme_status':'已验证，限本轮范围',
        'report':'../rby1-research/stage11/双臂分担与躯干协调_文献和仿真报告.md','results':'../rby1-research/stage11/summary.json',
        'verification':{'physical_runs':21,'quality_passes':21,'moving_runs':18,'braking_passes_moving':18,
                        'static_hold_runs':3,'grasp_passes_all_settings':17,'general_qp_vs_scipy_cases':32,
                        'box_kkt_cases':16,'half_timestep_methods':3,'default_exact_regression':True,'all_snapshot_hashes_verified':True},
        'findings':['固定姿态、相同物体总扳手下，内部分担对共同躯干的瞬时载荷映射约为零；臂部可变化。',
                    '偏心对称分担静止6秒时已有2.748毫米相对位移；QP和杠杆分担均约1.70毫米。不能把原失败全部称为急停特有问题。',
                    '偏心缓冲时几何杠杆分担与QP几乎相同，未证明复杂优化器必要或独立创新。',
                    '离线改变torso_2后，缓冲制动臂部归一化力矩30.11%降至24.40%，躯干35.81%升至45.34%，存在权衡。'],
        'qualification':'1 kg已知刚体、理想状态、6秒有限测试；曲线仍漂移，不是长期稳定、重载、全身主动优化或真机验证。',
        'next_section':{'status':'待开始','scheme_status':'待讨论','scope':'先定位静态及正常搬运可行而给定制动触及接触/关节能力的区间；必须比较几何分担和简单缓冲，再判断是否需主动躯干优化。'}})
    p['last_updated']='2026-10-31';dump(authority,p)
    print(json.dumps({'verified':True,'run_count':21,'default_exact':True,'archived_qp_compatibility_states':len(compat),'authority_updated':True}))


if __name__=='__main__':main()
