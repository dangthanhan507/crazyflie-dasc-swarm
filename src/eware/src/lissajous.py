##############################################
# lissajous.py
#
# Author: An Dang
#
#
# Description: This file contains the Lissajous
# trajectory class.
##############################################
import jax.numpy as jnp

class Lissajous:
    def __init__(self, a, omega, phi, off):
        self.a = a
        self.omega = omega
        self.phi = phi
        self.off = off
    def get_pos(self, t:float, D:int=0):
        
        if D == 0:
            x = self.a[0]*jnp.sin(self.omega[0]*t + self.phi[0]) + self.off[0]
            y = self.a[1]*jnp.sin(self.omega[1]*t + self.phi[1]) + self.off[1]
            z = self.a[2]*jnp.sin(self.omega[2]*t + self.phi[2]) + self.off[2]
        else:
            x = (self.omega[0]**D)*self.a[0]*jnp.sin(self.omega[0]*t + jnp.pi*D/2 + self.phi[0])
            y = (self.omega[0]**D)*self.a[1]*jnp.sin(self.omega[1]*t + jnp.pi*D/2 + self.phi[1])
            z = (self.omega[0]**D)*self.a[2]*jnp.sin(self.omega[2]*t + jnp.pi*D/2 + self.phi[2])
        return jnp.array([x,y,z])
    
    def get_desired_trajectory(self, t0: float, n_steps: int, dt:float, mass=0.05, g=9.81, traj_type: int = 1):
        xref = []
        
        # quat0 = jnp.array([1.0,0.0,0.0,0.0]).reshape((4,1))
        if traj_type == 1:
            for i in range(n_steps):
                ti = t0 + i*dt
                
                
                # xref_i = jnp.vstack((self.get_pos(ti).reshape((3,1)), quat0, self.get_pos(ti, D=1).reshape((3,1)), jnp.zeros((3,1))))
                xref_i = jnp.vstack((self.get_pos(ti).reshape((3,1)), jnp.zeros((3,1)), self.get_pos(ti, D=1).reshape((3,1)), jnp.zeros((3,1))))
                xref.append(xref_i)
        else:
            raise NotImplementedError
        
        xref = jnp.stack(xref).squeeze().T
        uref = jnp.zeros((4, n_steps-1)) + (mass*g/4)
        
        return xref, uref
    
    def get_desired_trajectory_point(self, t0: float, n_steps:int, dt:float, mass=0.05, g=9.81):
        xref = []
        for i in range(n_steps):
            ti = t0 + i*dt
            
            xref_i = jnp.array([[0.0,0.0,0.11,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]).T
            xref.append(xref_i)
        uref = jnp.zeros((4, n_steps-1)) + (mass*g/4)
        
        xref = jnp.stack(xref).squeeze().T
        
        return xref, uref
    
if __name__ == '__main__':
    A = jnp.array([5,5,0.7])
    omega = jnp.array([5/8, 4/8, 6/8])
    phi = jnp.array([jnp.pi/2, 0, 0])
    off = jnp.array([0,0,5])
    
    #3d plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    liss = Lissajous(A, omega, phi, off)
    # get points
    n_steps = 100
    t0 = 0
    dt = 0.1
    xref, uref = liss.get_desired_trajectory(t0, n_steps, dt)
    xref = np.array(xref)
    uref = np.array(uref)
    print(xref.shape, uref.shape)
    pts = xref[0:3,:]
    
    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pts[0,:], pts[1,:], pts[2,:])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()