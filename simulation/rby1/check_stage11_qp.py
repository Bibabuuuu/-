"""Independent SciPy comparison; run with the system Python, not the simulator."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from load_sharing import feasible_qp,check_solver,grasp_map,internal_basis


def main():
    rng=np.random.default_rng(923);rows=[]
    for n in [2,5]:
        for case in range(16):
            A=rng.normal(size=(n+3,n));H=A.T@A+.05*np.eye(n);g=rng.normal(size=n)*4
            C=np.r_[np.eye(n),-np.eye(n),rng.normal(size=(9,n))]
            d=np.r_[np.ones(2*n)*.6,rng.uniform(.1,1.2,9)]
            x,info=feasible_qp(H,g,C,d)
            other=minimize(lambda y:.5*y@H@y+g@y,np.zeros(n),jac=lambda y:H@y+g,
                           constraints={'type':'ineq','fun':lambda y:d-C@y,'jac':lambda y:-C},
                           method='SLSQP',options={'ftol':1e-12,'maxiter':300})
            error=float(np.max(np.abs(x-other.x)))
            assert info['accepted'] and other.success and error<2e-6,(info,other.message,error)
            rows.append({'n':n,'case':case,'solution_error_vs_scipy':error,**info})
    worst=0.
    for _ in range(30):
        com,left,right=rng.normal(size=(3,3));G=grasp_map(com,left,right);D=internal_basis(left,right)
        worst=max(worst,float(np.max(np.abs(G@D))))
    assert worst<1e-12
    out=Path(__file__).resolve().parents[2]/'outputs/rby1-research/stage11';out.mkdir(exist_ok=True)
    result={'box':check_solver(),'scipy_general_linear_comparisons':rows,
            'max_solution_error':max(r['solution_error_vs_scipy'] for r in rows),'max_grasp_nullspace_error':worst}
    (out/'qp_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['box','scipy_general_linear_comparisons']}))


if __name__=='__main__':main()
