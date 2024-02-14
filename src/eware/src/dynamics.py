##################################################
# dynamics.py
#
#
# Author: An Dang
#
# Description: This file contains the dynamics
# of the quadrotor and the linearization of the
# dynamics.
##################################################

import jax.numpy as jnp
import jax
import quat_utils as qu


class QuadRotor:
    def __init__(self):
        self.mass = 0.5
        self.leg_l = 0.1750
        self.J = jnp.diag(jnp.array([0.0023,0.0023, 0.004]))
        self.grav = 9.81
        self.kT = 1.0
        self.km = 0.0245
        self.fc = 0.1
        
        self.umin = jnp.array([[0.0,0.0,0.0,0.0]]).T + 0.2*self.mass*self.grav
        self.umax = jnp.array([[0.0,0.0,0.0,0.0]]).T + 0.6*self.mass*self.grav
        self.uhover = jnp.array([[0.0,0.0,0.0,0.0]]).T + 0.25*self.mass*self.grav
        
        self.thrust_max = 0.6*self.mass*self.grav
        self.thrust_min = 0.2*self.mass*self.grav
        self.thrust_hover = 0.25*self.mass*self.grav
        
    def quadTauB(self, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        '''
        Parameters
        ----------
        @param state: np.ndarray of size 13x1
        @param control: np.ndarray of size 4x1
        @param params: dict of parameters
        
        Description
        -----------
        Compute the moments in the body frame
        '''
        
        q = state[3:7]
        q = q/jnp.linalg.norm(q)
        
        rotB = jnp.array([[0,self.leg_l*self.kT,0,-self.leg_l*self.kT],
                          [-self.leg_l*self.kT,0,self.leg_l*self.kT,0],
                          [self.km,-self.km,self.km,-self.km]])
        
        tau = rotB@control
        return tau
    def quadFw(self, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        '''
        Parameters
        ----------
        @param state: np.ndarray of size 13x1
        @param control: np.ndarray of size 4x1
        @param params: dict of parameters
        
        Description
        -----------
        Compute the forces in the world frame
        '''
        
        q = state[3:7]
        q = q/jnp.linalg.norm(q)
        v = state[7:10]
        w = state[10:13]
        
        '''
            F^W = gravity + quadrotor thrust
            
            B = [0,0,0,0;
                 0,0,0,0;
                 1,1,1,1]; * kT
        '''
        Q = qu.Q(q)
        B = jnp.vstack((jnp.zeros((2,4)), jnp.ones((1,4)))) * self.kT
        F = (B@control)/self.mass - (qu.skew(w)@v)
        return F
    def quad_ode(self, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        '''
        Parameters
        ----------
        @param state: np.ndarray of size 13x1
        @param control: np.ndarray of size 4x1
        @param params: dict of parameters
        
        Description
        -----------
        Compute the dynamics of the quadrotor
        '''
        
        r = state[0:3]
        q = state[3:7]
        v = state[7:10]
        w = state[10:13]
        
        q = q/jnp.linalg.norm(q)
        
        Q = qu.Q(q)
        L = qu.L(q)
        H = jnp.vstack((jnp.zeros((1,3)), jnp.eye(3)))
        rdot = Q @ v
        qdot = 0.5 * L @ H @ w
        
        B = jnp.vstack((jnp.zeros((2,4)), self.kT * jnp.ones((1,4))))
        F = (B@control)/self.mass - (qu.skew(w)@v)
        vdot = F + Q.T @ jnp.array([[0,0,-self.grav]]).T
        
        rotB = jnp.array([[0,self.leg_l*self.kT,0,-self.leg_l*self.kT],
                          [-self.leg_l*self.kT,0,self.leg_l*self.kT,0],
                          [self.km,-self.km,self.km,-self.km]])
        tau = rotB@control
        
        wdot = jnp.linalg.inv(self.J) @ (-qu.skew(w) @ self.J @ w + tau)
        
        return jnp.vstack((rdot, qdot, vdot, wdot))
    def rk4(self, state: jnp.ndarray, control: jnp.ndarray, dt: float) -> jnp.ndarray:
        '''
        Parameters
        ----------
        @param state: np.ndarray of size 13x1
        @param control: np.ndarray of size 4x1
        @param dt: float
        
        Description
        -----------
        Compute the rk4 update of the quadrotor dynamics
        '''
        
        k1 = self.quad_ode(state, control)
        k2 = self.quad_ode(state + dt/2*k1, control)
        k3 = self.quad_ode(state + dt/2*k2, control)
        k4 = self.quad_ode(state + dt*k3, control)
        
        return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    def quad_linearize_discrete(self, state: jnp.ndarray, control: jnp.ndarray, dt: float) -> jnp.ndarray:
        '''
        Parameters
        ----------
        @param state: np.ndarray of size 13x1
        @param control: np.ndarray of size 4x1
        @param dt: float
        
        Description
        -----------
        Linearize the quadrotor dynamics using jax jacrev
        '''
        
        #linearize over quad_ode given control=self.uhover
        A = jax.jacrev(self.rk4, argnums=0)(state, control, dt).squeeze()
        B = jax.jacrev(self.rk4, argnums=1)(state, control, dt).squeeze()
        return A,B
    def quad_linearize_continuous(self, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        '''
        Parameters
        ----------
        @param state: np.ndarray of size 13x1
        @param control: np.ndarray of size 4x1
        
        Description
        -----------
        Linearize the quadrotor dynamics using jax jacrev
        '''
        
        #linearize over quad_ode given control=self.uhover
        A = jax.jacrev(self.quad_ode, argnums=0)(state, control).squeeze()
        B = jax.jacrev(self.quad_ode, argnums=1)(state, control).squeeze()
        return A,B
    def reduce_model(self, A, B, q0):
        QE = qu.E(q0)
        
        Abar = QE.T @ A @ QE
        Bbar = QE.T @ B
        return Abar, Bbar
    
    def reduce_state(self, state):
        # replace quaternion with rodrigues param
        xy0 = state[0:3]
        rp0 = qu.quat_to_rodparam(state[3:7])
        v0 = state[7:10]
        w0 = state[10:13]
        state = jnp.vstack((xy0, rp0, v0, w0))
        return state

if __name__ == '__main__':
    r0 = jnp.array([[0.0,0.0,1.0]]).T
    q0 = jnp.array([[1.0,0.0,0.0,0.0]]).T
    v0 = jnp.array([[0.0,0.0,0.0]]).T
    w0 = jnp.array([[0.0,0.0,0.0]]).T
    state0 = jnp.vstack((r0,q0,v0,w0))
    
    
    quad = QuadRotor()
    
    #test if we can fly it
    state = state0
    control = quad.uhover
    dt = 0.01
    for i in range(100):
        state = quad.rk4(state, control, dt)
    print(state)
    A,B = quad.quad_linearize_continuous(state0, quad.uhover)
    print(A.shape)
    print(B.shape)
    
    Ad, Bd = quad.quad_linearize_discrete(state0, quad.uhover, dt)
    print(Ad.shape)
    print(Bd.shape)
    
    Adbar, Bdbar = quad.reduce_model(Ad, Bd, q0)
    print(Adbar.shape)
    print(Bdbar.shape)
    
    Abar, Bbar = quad.reduce_model(A, B, q0)
    print(Abar.shape)
    print(Bbar.shape)
    
    C = B
    for i in range(13):
        C = jnp.hstack((C, jnp.linalg.matrix_power(A, i+1)@B))
    print(jnp.linalg.matrix_rank(C))
    
    #check if Abar and Bbar are controllable
    C = Bbar
    for i in range(12):
        C = jnp.hstack((C, jnp.linalg.matrix_power(Abar, i+1)@Bbar))
    print(jnp.linalg.matrix_rank(C))
    
    C = Bd
    for i in range(13):
        C = jnp.hstack((C, jnp.linalg.matrix_power(Ad, i+1)@Bd))
    print(jnp.linalg.matrix_rank(C))
    
    C = Bdbar
    for i in range(12):
        C = jnp.hstack((C, jnp.linalg.matrix_power(Adbar, i+1)@Bdbar))
    print(jnp.linalg.matrix_rank(C))