"""Explicit illustrative task limits; reference admission is not a safety proof."""
from stop_profiles import parameters, check_profile


def reference_options(acceleration_limit, clearance_m, *, speed=.4, base_T=.6,
                      stroke_cap=.09, acceleration_reserve=.02, space_reserve=.01, jerk=5.):
    A=acceleration_limit-acceleration_reserve
    assert A>0 and clearance_m>0
    lower=max(base_T,1.5*speed/A)
    upper=min(base_T+2*stroke_cap/speed,2*(clearance_m-space_reserve)/speed)
    p=parameters(speed,A,jerk)
    stroke=p['distance_m']-speed*base_T/2
    jerk_admitted=p['duration_s']>=base_T and stroke>=0 and stroke<=stroke_cap and p['distance_m']+space_reserve<=clearance_m
    if jerk_admitted:
        # Current forward-only IK path requires the relative velocity to stay nonnegative.
        try: check_profile(speed,A,jerk,base_T)
        except AssertionError: jerk_admitted=False
    return {'acceleration_limit_mps2':acceleration_limit,'clearance_m':clearance_m,
            'acceleration_reserve_mps2':acceleration_reserve,'space_reserve_m':space_reserve,
            'cubic':{'min_duration_s':lower,'max_duration_s':upper,'reference_admitted':lower<=upper+1e-12},
            'jerk_limited':{**p,'extra_stroke_m':stroke,'reference_admitted':bool(jerk_admitted)},
            'qualification':'Within these reference families and selected reserves only; orientation, contact, actual tracking and hardware need separate checks.'}


def assess(r,*,acceleration_limit,angle_limit_deg,clearance_m):
    observed=r['brake_distance_to_final_m'] is not None
    checks={'event_observed':observed,'physical_quality':bool(r['quality_passed']),
            'braking':bool(r['braking_passed']),'original_grasp':bool(r['original_grasp_criterion_passed']),
            'acceleration':observed and r['max_brake_payload_acceleration_mps2']<=acceleration_limit,
            'orientation':observed and r['max_brake_payload_target_angle_error_deg']<=angle_limit_deg,
            'clearance':observed and r['max_payload_front_travel_m']<=clearance_m}
    return {'limits':{'acceleration_mps2':acceleration_limit,'target_angle_deg':angle_limit_deg,'front_clearance_m':clearance_m},
            'checks':{k:bool(v) for k,v in checks.items()},'passed':all(checks.values()),
            'failed_checks':[k for k,v in checks.items() if not v],
            'scope':'Offline evaluation against illustrative task limits, not an online prediction or industrial requirement.'}
