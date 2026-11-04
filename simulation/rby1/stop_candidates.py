"""Finite mature-stop candidates and transparent tradeoffs; no new optimizer."""
import copy,time
from pathlib import Path
import numpy as np
from state_stop_reference import StateStop

ACCELERATIONS=(.5,.65,.78,1.,1.25)


def metrics(ref):
    z=ref.reference_samples;a=ref.audit
    row={'reference_acceleration_setting_mps2':ref.reference_acceleration,
         'kinematic_admission':a['preferred_plan_admitted'],'reasons':list(a['reasons']),
         'peak_reference_acceleration_mps2':float(max(abs(z[:,6]))),
         'duration_s':a['candidate_duration_s'],'relative_min_m':a['candidate_relative_min_m'],
         'relative_max_m':a['candidate_relative_max_m'],'predicted_envelope_m':a['candidate_envelope_front_travel_m'],
         'peak_reference_joint_speed_fraction':a['candidate_reference_joint_speed_fraction'],
         'normalized_upper_speed_square_integral_s':None,'joint_total_variation_rad':None}
    # Never extrapolate a rejected candidate beyond the checked interpolation.
    if a['needs_kinematic_preparation']:return row
    path=ref.path;length=path['extent']+path['backward_extent'];u=(z[:,7]+path['backward_extent'])/length
    q=np.polynomial.polynomial.polyval(u,path['coeff'])
    qv=np.polynomial.polynomial.polyval(u,path['first'])*((z[:,5]-z[:,2])/length)
    row['normalized_upper_speed_square_integral_s']=float(np.trapezoid(np.sum((qv/ref.vcaps[:,None])**2,axis=0),z[:,0]))
    row['joint_total_variation_rad']=float(np.sum(np.abs(np.diff(q,axis=1))))
    return row


def frontier_indices(rows):
    admitted=[i for i,r in enumerate(rows) if r['kinematic_admission']]
    def dominates(i,j):
        a,b=rows[i],rows[j]
        x=np.array([a['peak_reference_acceleration_mps2'],a['normalized_upper_speed_square_integral_s']])
        y=np.array([b['peak_reference_acceleration_mps2'],b['normalized_upper_speed_square_integral_s']])
        tolerance=np.array([1e-10,1e-12])
        return bool(np.all(x<=y+tolerance) and np.any(x<y-tolerance))
    return [i for i in admitted if not any(dominates(j,i) for j in admitted if i!=j)]


def choose(rows,priority):
    if priority not in ['acceleration','motion']:raise ValueError(priority)
    available=frontier_indices(rows)
    if not available:raise RuntimeError('No admitted candidate in this finite family; no task-safe fallback is asserted')
    order=['peak_reference_acceleration_mps2','normalized_upper_speed_square_integral_s']
    if priority=='motion':order.reverse()
    return min(available,key=lambda i:tuple(rows[i][k] for k in order)+(rows[i]['reference_acceleration_setting_mps2'],))


class CandidateStop(StateStop):
    """One-shot evaluation on the prepared posture path; explicit priority only."""
    def __init__(self,*,priority,accelerations=ACCELERATIONS,**kwargs):
        if priority not in ['acceleration','motion']:raise ValueError(priority)
        if not accelerations or not all(np.isfinite(a) and a>0 for a in accelerations):raise ValueError('Finite positive candidate A settings required')
        super().__init__(**kwargs)
        if self.strategy!='ruckig' or self.planning_delay!=0:raise ValueError('This offline prototype requires Ruckig and no emulated planning delay')
        self.priority=priority;self.accelerations=tuple(accelerations);self.selection_source_file=Path(__file__)

    def observe(self,t,m,d):
        if t>=self.trigger-1e-10 and not self.ready:
            start=time.perf_counter();refs=[];rows=[]
            for A in self.accelerations:
                ref=copy.copy(self);ref.audit=copy.deepcopy(self.audit);ref.events=[]
                ref.reference_acceleration=A;ref.audit['reference_acceleration_mps2']=A
                StateStop.observe(ref,t,m,d);refs.append(ref);rows.append(metrics(ref))
            selected=choose(rows,self.priority);preference=self.priority
            self.__dict__.update(refs[selected].__dict__)
            self.selection_audit={'priority':preference,'candidate_rows':rows,'frontier_indices':frontier_indices(rows),
                                  'selected_index':selected,'selected_A_mps2':self.reference_acceleration,
                                  'evaluation_and_selection_wall_ms':1000*(time.perf_counter()-start),
                                  'qualification':'Finite-family kinematic evaluation on the prepared posture path; command v/a seed for continuity. No global optimum, force feasibility, online timing guarantee or live-posture path rebuild.'}
            self.audit['candidate_selection']=copy.deepcopy(self.selection_audit);self.events[-1]=dict(self.audit)
            return
        StateStop.observe(self,t,m,d)
