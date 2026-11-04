"""Controlled forecast/execution separation, not a calibrated actuator fault model."""
from pathlib import Path
from state_stop_reference import LegacyStateStop as StateStop
from buffer_reference import stop


class BrakeMismatch(StateStop):
    def __init__(self,execution_T_s=.6,execution_start_delay_s=0.,**kwargs):
        super().__init__(seed_policy='command',batched=True,**kwargs)
        assert execution_T_s>0 and execution_start_delay_s>=0
        self.execution_T=execution_T_s;self.execution_delay=execution_start_delay_s
        self.execution_source_file=Path(__file__)

    def base_execution(self,t):
        if t<self.trigger:return self.prelude(t)
        x,v,_=self.prelude(self.trigger);elapsed=t-self.trigger
        if elapsed<=self.execution_delay:return x+v*elapsed,v,0.
        p,sv,a=stop(elapsed-self.execution_delay,v,self.execution_T)
        return x+v*self.execution_delay+p,sv,a

    def prepare(self,m,d,path):
        super().prepare(m,d,path)
        self.execution_config={'prediction_T_s':self.base_T,'execution_T_s':self.execution_T,
                               'execution_start_delay_s':self.execution_delay,
                               'execution_stop_s':self.trigger+self.execution_delay+self.execution_T,
                               'execution_brake_distance_m':self.speed_at_trigger*(self.execution_delay+self.execution_T/2),
                               'qualification':'The wheel controller tracks a separate prescribed execution curve. Object planning and arm-relative references retain the forecast. Future execution curve is not used by the planner; it is recorded for experiment auditing.'}
        self.audit['base_execution_emulation']=dict(self.execution_config)
