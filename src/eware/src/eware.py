import numpy as np
import jax.numpy as jnp

##############################################
# eware.py
#
# eware implementation from paper
# final product of this project
##############################################

'''
    given a reference path,
    Eware will generate a trajectory to follow
    ergodic trajectory.
    Eware will also generate a trajectory to follow
    back to home for charging.
    
    This back-to-base trajectory will be appended to the trajectory follower.
    
    If this appended trajectory is feasible, then we commit this.
    
    The idea is that we will keep committing trajectories.
    However, if the appended trajectory is not feasible, then no more 
    trajectories will be committed which will lead the drone back home.
    
    
'''
class Eware:
    def __init__(self):
        self.committed_trajectory = None
        pass