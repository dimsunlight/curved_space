import numpy as onp

from jax.config import config ; config.update('jax_enable_x64', True)
import jax.numpy as np
from jax import random
from jax import jit
from jax import lax
from jax import vmap
from jax import grad

import time

from jax_md import space, smap, energy, minimize, quantity, simulate, util

f32 = util.f32
f64 = util.f64
Array = util.Array

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d

from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from matplotlib import rc
rc('animation', html='jshtml')

def normalize(r1, Rad=1, **unused_kwargs):
  '''
  Normalize a vector r1 to lie on the sphere. 
    r1: the original position vector.
    Rad: the radius of the underlying sphere.
  Returns: the radius of the underlying sphere multiplied by the unit 
  vector of r1.
  '''
  return Rad*r1/np.linalg.norm(r1)

#essentially a function definition - just vmap normalization :)
batch_normalize = vmap(normalize,in_axes=[0, None])

def setup_sphere(Rad = 1):
    '''
    By defining everything in terms of spherical coordinates, we can generate
    displacements and shifts without regard to the underlying metric.
    Inputs:
        Rad: radius of underlying sphere
        ri: position vectors
        dr: shift amount
    '''
    
    '''
    def displacement_sphere_pair(r1,r2,**unused_kwargs):
        return r1 - r2
    '''
    def sphere_dist(r1,r2, Rad = Rad, **unused_kwargs):
        '''
        finds distance between two points on a sphere. 
        r1 : position in cartesian coordinates of first particle
        r2 : position in cartesian coordinates of second particle
        Rad : radius of underlying spherical geometry

        note: currently using most globally well-conditioned expression
        for finding the angle between two spherical points, not the most
        efficient.
        '''
        #make r1 and r2 unit vectors
        r1 = r1/np.linalg.norm(r1) 
        r2 = r2/np.linalg.norm(r2)
        magCross = np.sqrt(np.einsum("i,i",np.cross(r1,r2),np.cross(r1,r2)))
        magDot = np.einsum("i,i",r1,r2)
        return abs(Rad*np.arctan2(magCross,magDot)) #Retains quadrant information
    
    def shift_sphere_projection(r, dr, Rad = Rad, **unused_kwargs):
        '''
        Projection operator approach similar to Daniel's work on spheres. 

        Allows working in the tangent space -- of course, the limitation is 
        that motion in the tangent plane is only an approximation to motion on 
        the sphere. 
        '''
        rhat = batch_normalize(r,Rad) #rhat and dr are both (N,d)
        #calculation is dr_ij - r_ij (dr_ij rhat_kl) -> del(r)_ij. 
        firststep = np.einsum('ij,kl->ijkl',rhat,dr)
        proj_shift = dr - np.einsum('ij,ijkl->kl',rhat,firststep) #tensor contraction

        #newpos = r + proj_shift
        newpos = r - proj_shift #FOR SOME REASON, the force function is accidentally walking us *up* the energy gradient.
                                #Or there's something up with the shift function? This accomodates for that. 
        newpos = batch_normalize(newpos,Rad)

        return newpos
    
    return sphere_dist, shift_sphere_projection


def soft_sphere_simulation_force(metric, energy, N,
                                            sigma, alpha, epsilon,Rad=1):
  '''
  Initializes the force function for a soft sphere simulation. Piggybacks off
  of the JAX_MD definition of soft sphere energy with variables alpha,  
  epsilon, and sigma. 

  Inputs:
    metric: a function that returns the distance between two points in your 
      space.
    energy: function with returns the energy when fed the distance. Specifically
              a soft sphere energy utilizing three parameters sigma, alpha, and
              epsilon.
    sigma:  particle diameter
    alpha:  interaction strength parameter
    epsilon: interaction strength parameter
  Returns: 
    pairwise_forces, a function which calculates the total force on each 
    particle in the system. 
  '''

  sigma = sigma
  alpha = alpha
  epsilon = epsilon 

  energyGrad = grad(energy)
  metGrad = grad(metric)

  def indiv_force(r1,r2,**kwargs):
    '''
    force on r1 - vector output. 
    '''
    return -energyGrad(metric(r1,r2,Rad), sigma = sigma, alpha = alpha, epsilon = epsilon)*metGrad(r1,r2)

  def map_function(fxn,**kwargs):
    return vmap(vmap(fxn,(0,None),0),(None,0),0)

  def remove_diag(matrix,**kwargs):
    '''
    Right now, I'm using jax.numpy.nan_to_num because it should save computation time.
    So far, it doesn't seem to break, but I'm still suspicious.

    Also, big assumption here that diagonal elements are actually NaNs...
    '''
    matrix = np.nan_to_num(matrix) #changes nans to zero 
    mask = f32(1.0) - np.eye(N, dtype=matrix.dtype)
    if len(matrix.shape) == 3:
      mask = np.reshape(mask, (N, N, 1))
    return mask*matrix
    return matrix 

  def sum_rows(matrix,**kwargs):
    return np.einsum('ijk->jk',matrix)

  def pairwise_forces(R,**kwargs):
    '''
    Function that maps the individual forces to generate pairwise forces. Summing over row i
    yields the force on particle i.  
    '''
    fn_mapped = map_function(indiv_force) 
    forceMat = remove_diag(fn_mapped(R,R))
    return sum_rows(forceMat)
  
  return pairwise_forces
