"""Small load-sharing QP; fixed robot posture, no whole-body motion optimizer."""
import numpy as np
from payload_wrench import split_wrench


def skew(v):
    x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def internal_basis(left,right):
    # z: delta Fx, delta Fz, delta MLx/y/z, with normalized units.
    D=np.zeros((12,5));D[0,0]=1.;D[2,1]=1.;D[3:6,2:5]=np.eye(3)*.1
    D[6:9]=-D[:3];D[9:12]=-D[3:6]-skew(left-right)@D[:3]
    return D


def grasp_map(com,left,right):
    G=np.zeros((6,12));G[:3,:3]=G[:3,6:9]=np.eye(3)
    G[3:,:3]=skew(left-com);G[3:,6:9]=skew(right-com)
    G[3:,3:6]=G[3:,9:12]=np.eye(3)
    return G


def feasible_qp(H,g,C,d,tolerance=1e-9):
    """Primal feasible active-set solver for a strictly convex small QP."""
    x=np.zeros(len(g));active=[]
    if np.max(C@x-d)>1e-7:return x,{'accepted':False,'reason':'baseline violates allocator admission constraints'}
    for iteration in range(80):
        grad=H@x+g
        if active:
            W=C[active];K=np.block([[H,W.T],[W,np.zeros((len(active),len(active)))]])
            solution=np.linalg.lstsq(K,np.r_[-grad,np.zeros(len(active))],rcond=1e-12)[0]
            step=solution[:len(x)];lam=solution[len(x):]
        else:step=np.linalg.solve(H,-grad);lam=np.array([])
        if np.linalg.norm(step)<tolerance:
            if not active or min(lam)>=-1e-8:
                stationarity=np.linalg.norm(grad+(C[active].T@lam if active else 0),ord=np.inf)
                return x,{'accepted':True,'iterations':iteration+1,'max_constraint_violation':float(max(0,np.max(C@x-d))),
                          'stationarity_error':float(stationarity)}
            active.pop(int(np.argmin(lam)));continue
        rate=C@step;available=d-C@x;alpha=1.;blocking=None
        for i in range(len(d)):
            if i in active or rate[i]<=1e-11:continue
            candidate=max(0.,float(available[i]/rate[i]))
            if candidate<alpha:alpha=candidate;blocking=i
        x+=alpha*step
        if blocking is not None and alpha<1.-1e-10:active.append(blocking)
    return np.zeros(len(g)),{'accepted':False,'reason':'active-set iteration guard'}


def allocate(com,sites,grip_centers,F,M,B,tau_robot,caps,strategy='qp'):
    fs,ms=split_wrench(com,sites,F,M);w0=np.r_[fs['left'],ms['left'],fs['right'],ms['right']]
    D=internal_basis(sites['left'],sites['right']);T=B@D;t0=tau_robot+B@w0
    # Moment reference points are geometric handle centers, not the EE sites above them.
    P=np.zeros((6,12))
    for j,s in enumerate(['left','right']):
        k=6*j;P[3*j:3*j+3,k:k+3]=skew(sites[s]-grip_centers[s]);P[3*j:3*j+3,k+3:k+6]=np.eye(3)
    A=np.vstack([T/caps[:,None],P@D/.1,np.sqrt(.02)*np.eye(5)])
    b=np.r_[t0/caps,P@w0/.1,np.zeros(5)]
    H=A.T@A;g=A.T@b
    C=[T/caps[:,None],-T/caps[:,None],np.eye(5),-np.eye(5)]
    limits=np.array([1.5,1.5,3.,3.,3.])
    rhs=[1.-t0/caps,1.+t0/caps,limits,limits]
    # Conservative tangential-force diamond, assumed total normal 8.75 N per hand.
    for j in [0,1]:
        k=6*j
        for sx in [-1.,1.]:
            row=np.zeros(12);row[k]=sx;row[k+2]=1.
            C.append((row@D)[None,:]);rhs.append(np.array([7.-row@w0]))
        row=np.zeros(12);row[k+2]=-1.
        C.append((row@D)[None,:]);rhs.append(np.array([row@(-w0)]))
    C=np.vstack(C);rhs=np.concatenate(rhs)
    if strategy=='qp':z,info=feasible_qp(H,g,C,rhs)
    elif strategy=='lever':
        # Pure geometric baseline: share by the COM projection along the handle line.
        r=grip_centers['left']-grip_centers['right']
        alpha=float(np.clip((com-grip_centers['right'])@r/(r@r),0.,1.))
        fl=alpha*F;fr=(1-alpha)*F
        residual=M-np.cross(grip_centers['left']-com,fl)-np.cross(grip_centers['right']-com,fr)
        ml=residual/2-np.cross(sites['left']-grip_centers['left'],fl)
        mr=residual/2-np.cross(sites['right']-grip_centers['right'],fr)
        target=np.r_[fl,ml,fr,mr];z=np.linalg.lstsq(D,target-w0,rcond=None)[0]
        assert np.max(np.abs(w0+D@z-target))<1e-9
        violation=float(max(0,np.max(C@z-rhs)))
        info={'accepted':violation<1e-7,'max_constraint_violation':violation,'left_fraction':alpha}
        if not info['accepted']:z=np.zeros(5);info['reason']='geometric baseline violates shared admission constraints'
    else:raise ValueError(strategy)
    w=w0+D@z
    info.update({'objective_before':float(b@b),'objective_after':float(np.linalg.norm(A@z+b)**2),
                 'max_wrench_balance_error':float(np.max(np.abs(grasp_map(com,sites['left'],sites['right'])@w-np.r_[F,M]))),
                 'desired_left_vertical_N':float(w[2]),'desired_right_vertical_N':float(w[8]),
                 'max_grip_center_moment_Nm':float(np.max(np.abs(P@w)))})
    return {'left':w[:3],'right':w[6:9]},{'left':w[3:6],'right':w[9:12]},info


def check_solver():
    rng=np.random.default_rng(817);rows=[]
    for n in [2,5]:
        for k in range(8):
            A=rng.normal(size=(n+2,n));H=A.T@A+.1*np.eye(n);g=rng.normal(size=n)*3
            C=np.r_[np.eye(n),-np.eye(n)];d=np.ones(2*n)*.4
            x,info=feasible_qp(H,g,C,d);assert info['accepted'] and info['max_constraint_violation']<1e-7 and info['stationarity_error']<1e-7
            # Coordinate KKT signs for the box, independently of active-set multipliers.
            grad=H@x+g
            assert all(abs(grad[i])<1e-7 if abs(x[i])<.4-1e-7 else (grad[i]<=1e-7 if x[i]>.0 else grad[i]>=-1e-7) for i in range(n))
            rows.append({'n':n,'iterations':info['iterations']})
    return {'box_kkt_cases':rows,'passed':True}
