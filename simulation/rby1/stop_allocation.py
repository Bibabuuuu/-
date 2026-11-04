"""Matched object stops with different base profiles; no joint optimizer."""
from pathlib import Path
from state_stop_reference import StateStop,make_otg

CASES={'base_only':(.5,.5),'shared':(.65,.5),
       'base_fast_buffer':(1.,.5),'base_fast_held':(1.,1.)}


class AllocationStop(StateStop):
    def __init__(self,case):
        self.case=case;self.base_A,object_A=CASES[case]
        self.base_trajectory=make_otg(.4,0.,self.base_A,5.)
        super().__init__(speed=.4,base_T=self.base_trajectory.duration,seed_policy='command',
                         batched=True,reference_acceleration=object_A,reference_jerk=5.)
        self.task_source_file=Path(__file__)

    def base(self,t):
        if t<self.trigger:return self.prelude(t)
        x=self.prelude(self.trigger)[0]
        p,v,a=self.base_trajectory.at_time(min(max(0.,t-self.trigger),self.base_trajectory.duration))
        return x+p[0],v[0],a[0]

    def prepare(self,m,d,path):
        super().prepare(m,d,path)
        self.audit['base_object_allocation']={'case':self.case,'base_A_setting_mps2':self.base_A,
            'base_J_setting_mps3':5.,'base_profile':'Ruckig velocity OTG',
            'base_duration_s':self.base_T,'base_reference_distance_m':self.speed*self.base_T/2,
            'same_world_x_feedback':True,'external_obstacle_in_model':False,
            'qualification':'Shape parameters, not hardware maxima. Zero nominal upper stroke retains the same measured-base tracking corrections. No joint optimality or environmental collision certificate.'}
