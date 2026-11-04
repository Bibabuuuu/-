"""Independent forward delivery, followed by continuous mature stopping."""
from pathlib import Path
import numpy as np
from state_stop_reference import StateStop


def delivery(t,start,duration=1.2,stroke=.04):
    u=(t-start)/duration
    if u<=0:return (0.,0.,0.)
    if u>=1:return (stroke,0.,0.)
    return (stroke*(10*u**3-15*u**4+6*u**5),
            stroke/duration*30*u*u*(1-u)**2,
            stroke/duration**2*60*u*(1-u)*(1-2*u))


class DeliveryStop(StateStop):
    def __init__(self,phase,*,delivery_duration=1.2,delivery_stroke=.04,**kwargs):
        if not np.isfinite(phase) or not 0<phase<1:raise ValueError('Trigger phase must be inside the delivery motion')
        if not np.isfinite(delivery_duration) or delivery_duration<=0 or not np.isfinite(delivery_stroke) or delivery_stroke<0:raise ValueError('Finite positive duration and nonnegative delivery stroke required')
        super().__init__(**kwargs)
        if self.strategy!='ruckig' or self.seed_policy!='command' or self.planning_delay!=0:raise ValueError('Delivery handover prototype uses continuous command seeds, Ruckig and no emulated planning delay')
        self.delivery_duration=delivery_duration;self.delivery_stroke=delivery_stroke;self.phase=phase
        self.delivery_start=self.trigger-phase*delivery_duration;self.task_source_file=Path(__file__)

    def delivery(self,t):return delivery(t,self.delivery_start,self.delivery_duration,self.delivery_stroke)

    def object_prelude(self,t):
        b=self.prelude(t);r=self.delivery(t)
        return tuple(b[i]+r[i] for i in range(3))

    def prepare(self,m,d,path):
        super().prepare(m,d,path)
        self.audit['delivery_task']={'start_s':self.delivery_start,'duration_s':self.delivery_duration,'stroke_m':self.delivery_stroke,
                                     'trigger_progress':self.phase,'relative_reference_at_trigger':list(self.delivery(self.trigger)),
                                     'object_reference_at_trigger':list(self.object_prelude(self.trigger)),
                                     'qualification':'Independent pre-stop task motion; no real hardware speed or payload damage calibration.'}
