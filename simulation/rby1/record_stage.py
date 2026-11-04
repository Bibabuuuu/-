"""Verify evidence integrity and update the existing authoritative project record."""
from pathlib import Path
import hashlib,json,re
from build_model import ROOT,OUT,dump

def main():
    suite=json.loads((OUT.parent/'base_suite.json').read_text(encoding='utf-8'))
    params=json.loads((OUT.parent/'parameter_validation.json').read_text(encoding='utf-8'))
    digest=hashlib.sha256((OUT/'research_model.xml').read_bytes()).hexdigest()
    assert suite['passed'] and params['passed']
    assert all(r['model_sha256']==digest for r in suite['results'])
    assert params['source_model_sha256']==digest
    assert all(r['script_sha256']==hashlib.sha256((ROOT/'simulation/rby1/run_base.py').read_bytes()).hexdigest() for r in suite['results'])
    source=ROOT/'work/robot-model-audit/sources/rby1-pinned'
    manifest=json.loads((source/'download_manifest.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((source/e['path']).read_bytes()).hexdigest()==e['sha256'] for e in manifest['files'])
    report=ROOT/'docs/plans/02-rby1-research-base.md'
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
        if not target.startswith(('http://','https://')):
            assert (report.parent/target).resolve().exists(),target
    path=ROOT/'outputs/robot-topic-study/双课题范围更新.json'
    record=json.loads(path.read_text(encoding='utf-8'))
    record['last_updated']='2026-10-31'
    record['active_focus']['next_phase_status']='RB-Y1名义参数统一与空载底盘验证已完成；下一小节为刚体载荷的双臂接触抓持，状态待开始。正常任务与VLA不在当前实施范围。'
    record['active_focus']['technical_choices']='已按用户继续指令采用RB-Y1作为当前仿真原型；MuJoCo 3.17.0、URDF统一名义参数和显式轮地/执行器假设已实现。硬件型号、载荷任务、完整急停控制算法及真机接口仍未最终确认。'
    direction=record['direction_a']
    direction['implementation_status']='已实现RB-Y1派生研究模型、参数一致性检查和有界空载底盘基线；5姿态参数对照与6项动态工况通过。MATLAB脚本未执行，载荷抓持、NMPC/WBC与急停持物效果未验证。'
    direction['model_selection_audit']['next_section']={
        'status':'已完成','plan':'../../docs/plans/02-rby1-research-base.md',
        'scope':'名义参数统一、能力约束和空载直行制动验证；未包含载荷抓持'}
    direction['rby1_research_model_stage']={
        'date':'2026-10-31','status':'已完成','authorization':'用户在前一轮推荐RB-Y1及实施顺序后回复“继续”',
        'plan':'../../docs/plans/02-rby1-research-base.md','code':'../../simulation/rby1',
        'model':'../rby1-research/model/research_model.xml','parameters':'../rby1-research/model/parameters.json',
        'validation':'../rby1-research/parameter_validation.json','dynamic_suite':'../rby1-research/base_suite.json',
        'preview':'../rby1-research/base_braking_preview.gif',
        'result_summary':{'parameter_poses_passed':5,'empty_dynamic_cases_passed':6,
                          'nominal_brake_from_mps':.25,'nominal_distance_to_speed_threshold_m':suite['results'][0]['signed_brake_distance_to_settled_m'],
                          'stop_speed_threshold_mps':.01,'half_timestep_distance_difference_m':suite['half_timestep_brake_distance_delta_m']},
        'qualification':'仅规定名义模型与低速空载工况；理想被动支撑、执行器附加惯量/阻尼与部分力矩限值是假设，不是真机标定。',
        'matlab_status':'已提供导入与对照脚本，当前环境未发现可调用MATLAB，尚未执行。',
        'next_section':{'status':'待开始','scope':'刚体载荷的双臂静态接触抓持及制动基线；不将物体焊接代替抓持验证'}}
    dump(path,record)
    dump(OUT.parent/'stage_integrity.json',{'model_sha256':digest,'parameter_and_dynamic_evidence_match_current_model':True,
         'all_source_files_unchanged':len(manifest['files']),'report_local_links_valid':True,
         'git_status':'Task workspace has no Git repository; no commit performed',
         'matlab_execution':'not performed'})
    print('Stage record updated; model evidence, source integrity and report links verified.')

if __name__=='__main__':main()
