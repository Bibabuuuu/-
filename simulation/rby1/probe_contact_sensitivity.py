"""Isolated scene variants for numerical sensitivity, never the default model."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import run_payload as baseline
from build_model import dump
from probe_boundary import diagnostics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('probe_folder', type=Path)
    args=p.parse_args()
    source=baseline.PAY
    source_hash=hashlib.sha256((source/'payload_scene.xml').read_bytes()).hexdigest()
    results=[]
    for name,impratio,solref in [('impratio20',20,None),('solref4ms',None,.004)]:
        scene=args.probe_folder/'sensitivity_scenes'/name
        scene.mkdir(parents=True,exist_ok=True)
        tree=ET.parse(source/'payload_scene.xml')
        if impratio is not None: tree.getroot().find('option').set('impratio',str(impratio))
        if solref is not None:
            for g in tree.getroot().iter('geom'):
                n=g.get('name','')
                if n.endswith('_pad') or n.startswith('payload_'):g.set('solref',f'{solref} 1')
        tree.write(scene/'payload_scene.xml',encoding='utf-8',xml_declaration=True)
        init=json.loads((source/'initialization.json').read_text())
        init['scene_sha256']=hashlib.sha256((scene/'payload_scene.xml').read_bytes()).hexdigest()
        init['sensitivity_note']='Same qpos and geometry as the original; only the recorded contact numerical setting differs.'
        dump(scene/'initialization.json',init)
        overlay={'source_scene_sha256':source_hash,'impratio_override':impratio,'pad_payload_solref_time_constant_s_override':solref}
        dump(scene/'override.json',overlay)
        # Select an isolated input fixture for the unchanged sequential runner.
        # This does not alter qpos/qvel during physical integration.
        try:
            baseline.PAY=scene.resolve()
            for phase,speed in [('static',0.),('brake',.4)]:
                r=diagnostics(baseline.run(label=f'b04_sensitivity_{name}_{phase}',mass=1.35,speed=speed,brake_duration=.3))
                r['contact_override']=overlay
                dump(Path(r['folder'])/'contact_override.json',overlay)
                results.append(r)
                dump(args.probe_folder/'contact_sensitivity.json',{'cases':results,'reason':'Test dependence on uncalibrated soft-contact parameters; no retuning of default model or task threshold.'})
        finally:
            baseline.PAY=source
    assert hashlib.sha256((source/'payload_scene.xml').read_bytes()).hexdigest()==source_hash
    (args.probe_folder/'contact_sensitivity_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
