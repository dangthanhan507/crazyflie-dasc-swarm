import control as ct
from lissajous import Lissajous
from dynamics import QuadRotor
import jax.numpy as jnp
import cvxpy as cp
import numpy as np
import scipy.linalg as splg
import quat_utils as qu
class LQR:
    def __init__(self, Q, R, A, B):
        self.Q = Q
        self.R = R
        self.A = A
        self.B = B
        
        self.K = ct.lqr(A, B, Q, R)[0]
    def get_control(self, x):
        return -self.K @ x
    
# There are two MPC's. One for back-to-base trajectory generation and other for following a path.
class Controller:
    def __init__(self, lissajous_traj: Lissajous, Q, R, Ad, Bd, x0, quad: QuadRotor):
        self.lissajous = lissajous_traj
        
        #cost matrices
        self.Q = Q
        self.R = R
        
        #dynamic matrices (discrete) assumed to be lienarized about an eq pt
        self.A = Ad
        self.B = Bd
        
        #the eq pt
        self.xeq = x0
        
        self.q0 = x0[3:7] #quat0
        
        self.quad = quad
        
        pass
    def get_nominal_mpc_policy(self, x0: jnp.array, t0=0.0, n_steps:int=40, dt=0.01, mpc_type="NM"):
        '''
        Parameters
        ----------
        @param x0: np.ndarray of size 13x1
        @param t0: float
        @param n_steps: int
        @param TH_sec: float
        @param mpc_type: str
            -> NM: Nominal MPC points for state and control
            -> B2B: Back-to-base MPC points for state and control
        '''
        
        
        #xref assumed in 12 dim
        if mpc_type=="NM":
            xref, uref = self.lissajous.get_desired_trajectory(t0, n_steps, dt)
        else:
            xref, uref = self.lissajous.get_desired_trajectory_point(t0, n_steps, dt)
        
        #make everything reduce order
        Ad, Bd = self.quad.reduce_model(self.A, self.B, self.q0)
        
        
        eq_x = np.array(self.quad.reduce_state(self.xeq)).flatten()
        eq_u = np.array(self.quad.uhover).flatten()
        
        
        xref = np.array(xref)
        uref = np.array(uref)
        
        # change x0 to be shape 12
        x0 = self.quad.reduce_state(x0)
        
        Ad = np.array(Ad)
        Bd = np.array(Bd)
        xvar = cp.Variable((x0.shape[0], n_steps))
        uvar = cp.Variable((4, n_steps-1))
        
        cost = 0.0
        constr = [xvar[:,0] == x0.flatten()]
        ineq_constr = []
        for t in range(n_steps-1):
            cost += cp.quad_form(xvar[:,t] - xref[:,t], self.Q, assume_PSD=True) + cp.quad_form(uvar[:,t] - uref[:,t], self.R, assume_PSD=True)
            constr += [xvar[:,t+1] == eq_x + Ad @ (xvar[:,t] - eq_x) + Bd @ (uvar[:,t] - eq_u)]
            ineq_constr += [uvar[:,t] <= self.quad.thrust_max, uvar[:,t] >= self.quad.thrust_min]
            
        prob = cp.Problem(cp.Minimize(cost), constr + ineq_constr)
        #print problem type is QP
        # print(prob.is_qp()) #printed True
        prob.solve()
        
        #print cost
        print('Cost: ', prob.value)
        
        
        ts = np.arange(n_steps)*dt
        
        xvar = np.array(xvar.value)
        uvar = np.array(uvar.value)
        
        return ts, xvar, uvar
    
if __name__ == '__main__':
    #initial state
    t0 = 0
    r0 = jnp.array([[0.0,0.0,1.0]]).T
    q0 = jnp.array([[1.0,0.0,0.0,0.0]]).T
    v0 = jnp.array([[0.0,0.0,0.0]]).T
    w0 = jnp.array([[0.0,0.0,0.0]]).T
    state0 = jnp.vstack((r0,q0,v0,w0))
    
    #example lissajous trajectory
    A = jnp.array([0.5,0.5,0.3])
    omega = jnp.array([5/8, 4/8, 6/8])
    phi = jnp.array([jnp.pi/4, 0, 0])
    off = jnp.array([0,0,1])
    
    liss = Lissajous(A, omega, phi, off)
    
    # [5,0,5] is start
    quad = QuadRotor()
    dt = 0.5
    n_steps = 100
    Ad, Bd = quad.quad_linearize_discrete(state0, quad.uhover, dt) #linearize
    
    #mpc on reduced model
    
    #12 states
    # Q = 1e3*jnp.diag(jnp.array([1.0,1.0,1.0,0.0,0.0,0.0,1e-2,1e-2,1e-2,0,0,0]))
    Q = 100*jnp.eye(12)
    
    #13 states
    # Q = 1e3*jnp.diag(jnp.array([1.0,1.0,1.0,0.0,0.0,0.0,0.0,1e-2,1e-2,1e-2,0,0,0]))
    
    R = 1*jnp.eye(4)
    ctrler = Controller(liss, Q, R, Ad, Bd, state0, quad)
    
    
    xref, uref = liss.get_desired_trajectory(t0, n_steps, dt)
    # xref, uref = liss.get_desired_trajectory_point(t0, n_steps, dt)
    
    ts, xvar, uvar = ctrler.get_nominal_mpc_policy(state0, dt=dt, mpc_type="NM")
    
    print(xref[0,:])
    print(xref[1,:])
    print(xref[2,:])
    
    #plot xvar in 3d
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(xvar[0,:], xvar[1,:], xvar[2,:], label='mpc curve')
    ax.plot(xref[0,:], xref[1,:], xref[2,:], 'o', label='ref curve')
    #plot state0
    ax.plot(state0[0], state0[1], state0[2], 'o', label='start')
    ax.legend()
    
    
    #2d plot of uvar vs time
    plt.figure()
    plt.plot(ts[:-1], uvar[0,:], label='u1')
    plt.plot(ts[:-1], uvar[1,:], label='u2')
    plt.plot(ts[:-1], uvar[2,:], label='u3')
    plt.plot(ts[:-1], uvar[3,:], label='u4')
    plt.legend()
    plt.show()
    
    # save npy
    np.save('xvar.npy', xvar)
    pass