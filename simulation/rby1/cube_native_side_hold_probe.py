"""Prototype a 250 mm side hold using the original hand collision meshes."""
import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from build_model import OUT, dump
from build_payload import solve_ik
import cube_broad_pad_prototype as continuous
from cube_grasp_probe import read_json
from motion_distribution import nonpad_penetration
from payload_wrench import split_wrench

SOURCE = OUT.parent / "s3-cube-broad-pad-v1"
DEST = OUT.parent / "s3-cube-native-side-hold-v1"
HALF_EDGE = .125


def prepare(forward_clearance, flat_finger):
    DEST.mkdir(exist_ok=True)
    assert not (DEST / "scene.xml").exists()
    tree = ET.parse(SOURCE / "scene.xml")
    original = ET.parse(OUT / "research_model.xml")
    asset = tree.find("asset")
    mesh_names = {mesh.get("name") or Path(mesh.get("file")).stem for mesh in asset.findall("mesh")}
    native_geometry = {}
    for side in ("left", "right"):
        wrist = tree.find(f".//body[@name='ee_{side}']")
        for geom in wrist.findall("geom"):
            if geom.get("name").endswith(("_broad_pad", "_pad_mount")):
                wrist.remove(geom)
        # Identify original housing pieces as grasp contacts in the observer.
        for geom in wrist.findall("geom[@class='collision']"):
            old = geom.get("name")
            new = f"ee_finger_{side[0]}_native_housing_{len(native_geometry)}_pad"
            geom.set("name", new)
            native_geometry[new] = old
        for finger in wrist.findall("body"):
            name = finger.get("name")
            if not name.startswith("ee_finger_"):
                continue
            finger.remove(finger.find(f"geom[@name='{name}_pad']"))
            body = original.find(f".//body[@name='{name}']")
            for geom in body.findall("geom[@class='collision']"):
                current = deepcopy(geom)
                old = geom.get("name")
                new = old + "_native_pad"
                current.set("name", new)
                native_geometry[new] = old
                mesh_name = geom.get("mesh")
                assert mesh_name is None or mesh_name in mesh_names, mesh_name
                finger.append(current)
    for geom in tree.iter("geom"):
        if geom.get("name") in native_geometry:
            geom.set("friction", ".8 0 0")
            geom.set("condim", "3")
            geom.set("priority", "1")
            geom.set("solref", ".008 1")
    scene = DEST / "scene.xml"
    tree.write(scene, encoding="utf-8", xml_declaration=True)
    model = mj.MjModel.from_xml_path(str(scene))
    data = mj.MjData(model)
    with np.load(SOURCE / "initial.npz", allow_pickle=False) as saved:
        seed = saved["qpos"].copy()
    for suffix in ("l1", "r1", "l2", "r2"):
        joint = model.joint("gripper_finger_" + suffix)
        seed[joint.qposadr[0]] = -.003 if suffix.endswith("1") else .003
    data.qpos[:] = seed
    mj.mj_kinematics(model, data)
    cube = model.geom("payload_cube").id
    center = data.geom_xpos[cube].copy()
    orientation = data.geom_xmat[cube].reshape(3, 3).copy()
    center += forward_clearance * orientation[:, 0]
    cube_q = int(model.joint("payload_free").qposadr[0])
    vertices = {}
    for i in range(model.ngeom):
        if model.geom_type[i] == mj.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[i]
            start = model.mesh_vertadr[mesh]
            vertices[i] = model.mesh_vert[start:start+model.mesh_vertnum[mesh]].astype(float)
    local = {}
    contact_ids = {}
    for side in ("left", "right"):
        site = model.site("ee_"+side+"_site").id
        wrist = data.site_xpos[site]
        R = data.site_xmat[site].reshape(3, 3)
        contact_ids[side] = [model.geom(n).id for n in native_geometry if n.startswith(f"ee_finger_{side[0]}")]
        ids = [i for i in vertices if i in contact_ids[side] or (
            model.geom_group[i] == 2 and model.geom(i).name.startswith((f"ee_{side}_", f"ee_finger_{side[0]}")))]
        local[side] = np.vstack([(vertices[i] @ data.geom_xmat[i].reshape(3, 3).T +
                                  data.geom_xpos[i] - wrist) @ R for i in ids])
    attempts = []
    offsets = None
    candidates = (product((.045, .025), (75., 90., 60.), (.215, .205, .195))
                  if flat_finger else product((-.035, -.012), (38., 32., 44., 24.), (0., .025)))
    for wrist_x, angle_deg, height in candidates:
        data.qpos[:] = seed
        data.qpos[cube_q:cube_q+3] = center
        fits = []
        for side, sign in (("left", 1.), ("right", -1.)):
            angle = np.deg2rad(angle_deg)
            if flat_finger:
                # The finger broad face has its normal along local x.
                # Pitch about that normal so the housing clears the top edge.
                xaxis = np.array([0., sign, 0.])
                zaxis = np.array([-np.cos(angle), 0., np.sin(angle)])
            else:
                xaxis = np.array([np.sin(angle), sign*np.cos(angle), 0.])
                zaxis = np.array([-np.cos(angle), sign*np.sin(angle), 0.])
            rotation = np.column_stack((xaxis, np.cross(zaxis, xaxis), zaxis))
            target = np.array([wrist_x, 0., height])
            points = local[side] @ rotation.T + target
            inside = (abs(points[:, 0]) <= HALF_EDGE+.001) & (abs(points[:, 2]) <= HALF_EDGE+.001)
            target[1] = sign * (HALF_EDGE - (sign*points[inside, 1]).min() + .00025)
            fits.append(solve_ik(model, data, side, center+orientation@target, orientation@rotation))
        distance = {}
        points = {}
        for side, ids in contact_ids.items():
            candidates = []
            for i in ids:
                witness = np.zeros(6)
                gap = mj.mj_geomDistance(model, data, i, cube, 2., witness)
                candidates.append((gap, i, witness))
            gap, i, witness = min(candidates, key=lambda item:item[0])
            distance[side] = dict(gap_m=float(gap), native_geometry=native_geometry[model.geom(i).name],
                                  cube_point_m=witness[3:].tolist())
            points[side] = witness[3:].copy()
        error = max(f["position_error_m"] for f in fits)
        angle = max(f["angle_error_rad"] for f in fits)
        undesired = nonpad_penetration(model, data)
        visible = max(0., *(float((HALF_EDGE - abs((vertices[i] @ data.geom_xmat[i].reshape(3, 3).T
                         + data.geom_xpos[i] - center) @ orientation)).min(axis=1).max())
                         for i in vertices if model.geom_group[i] == 2))
        attempts.append(dict(wrist_x_m=wrist_x,rotation_deg=angle_deg,height_m=height,fit_error_m=error,
            fit_angle_error_rad=angle,unexpected_penetration_m=undesired,visible_mesh_intrusion_m=visible,
            native_contacts=distance))
        on_finger = all("EE_FINGER_collision_2" in c["native_geometry"] for c in distance.values())
        if (not flat_finger or on_finger) and error < 1e-4 and angle < .001 and undesired < 1e-5 and visible < 1e-5 and all(
                -.00005 <= c["gap_m"] <= .0015 for c in distance.values()):
            offsets = {side: (points[side]-data.site_xpos[model.site("ee_"+side+"_site").id]).tolist() for side in points}
            np.savez(DEST / "initial.npz", qpos=data.qpos, qvel=np.zeros(model.nv))
            break
    dump(DEST / "preflight.json", dict(passed=offsets is not None, attempts=attempts,
         contact_offsets_world_m=offsets,native_geometry=native_geometry,
         target_normal_per_hand_N=10., extra_forward_clearance_m=forward_clearance, flat_finger=flat_finger,
         normal_force_assumption="MuJoCo nominal contact; no add-on parts"))
    print("geometry",offsets is not None,attempts[-1],flush=True)


def execute(phase):
    preflight = read_json(DEST / "preflight.json")
    assert preflight["passed"]
    offsets = {s:np.array(p) for s,p in preflight["contact_offsets_world_m"].items()}

    def contact_wrench(com, wrists, force, moment):
        centers = {side: wrists[side]+offset for side,offset in offsets.items()}
        forces, moments = split_wrench(com, centers, force, moment)
        normal = centers["right"]-centers["left"]
        normal /= np.linalg.norm(normal)
        for side, sign in (("left",1.),("right",-1.)):
            forces[side] += sign * 10. * normal
            moments[side] += np.cross(offsets[side], forces[side])
        return forces, moments

    with patch.multiple(continuous, DEST=DEST, pad_wrench=contact_wrench):
        if phase == "static":
            continuous.static()
        else:
            continuous.brake()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","static","brake"))
    parser.add_argument("--forward-clearance",type=float,choices=(0.,.04),default=0.)
    parser.add_argument("--flat-finger",action="store_true")
    args=parser.parse_args()
    if args.flat_finger:
        DEST=OUT.parent / "s3-cube-native-flat-finger-v1"
    elif args.forward_clearance:
        DEST=OUT.parent / "s3-cube-native-side-hold-v2"
    if args.phase=="prepare":
        prepare(args.forward_clearance,args.flat_finger)
    else:
        execute(args.phase)
