##############################################
# Quaternion utilities
# quat_utils.py
# 
# Author: An Dang
#
# Description: This file contains utilities 
# for working with quaternions.
##############################################

import jax.numpy as jnp
import jax.scipy as jsp

QUAT_H = jnp.vstack((jnp.zeros((1,3)), jnp.eye(3)))
QUAT_T = jnp.diag(jnp.array([1,-1,-1,-1]))

def skew(x: jnp.ndarray) -> jnp.ndarray:
    '''
    Parameters
    ----------
    @param x: jnp.ndarray of size 3x1
    @return skew(x): jnp.ndarray of size 3x3
    '''
    xp = x.flatten()
    return jnp.array([[0,-xp[2],xp[1]],
                     [xp[2],0,-xp[0]],
                     [-xp[1],xp[0],0]])
    
def L(q: jnp.ndarray) -> jnp.ndarray:
    '''
    Parameters
    ----------
    @param q: jnp.ndarray of size 4x1
    @return L(q): jnp.ndarray of size 4x4
    '''
    
    qv = q.flatten()
    qw = qv[0]
    qx = qv[1]
    qy = qv[2]
    qz = qv[3]
    
    Lmat = jnp.array([[qw,-qx,-qy,-qz],
                  [qx, qw,-qz, qy],
                  [qy, qz, qw,-qx],
                  [qz,-qy, qx, qw]])
    
    return Lmat
def Q(q: jnp.ndarray) -> jnp.ndarray:
    '''
    Parameters
    ----------
    @param q: jnp.ndarray of size 4x1
    '''
    
    H = QUAT_H
    T = QUAT_T
    
    return H.T @ T @ L(q) @ T @ L(q) @ H

def G(q: jnp.ndarray) -> jnp.ndarray:
    '''
    Parameters
    ----------
    @param q: jnp.ndarray of size 4x1
    '''
    H = QUAT_H
    return L(q) @ H

def E(q: jnp.ndarray) -> jnp.ndarray:
    '''
    Parameters
    ----------
    @param q: jnp.ndarray of size 4x1
    '''
    return jsp.linalg.block_diag(1.0*jnp.eye(3), G(q), 1.0*jnp.eye(6))


def quat_to_rodparam(q: jnp.ndarray):
    return q[1:4]/q[0]

def rodparam_to_quat(r: jnp.ndarray):
    normalizing = 1/(jnp.sqrt(1 + jnp.dot(r,r)))
    return normalizing * jnp.array([1, r[0], r[1], r[2]])