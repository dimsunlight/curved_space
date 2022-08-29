import numpy as onp

import jax
from jax.config import config ; config.update('jax_enable_x64', True)
import jax.numpy as np
from jax.numpy import cos, sin, sqrt
from jax import vmap
from jax import grad
from jax_md import util
from jax_md.colab_tools import renderer

f32 = util.f32
f64 = util.f64

def hav(theta):
  return (np.sin(theta/2))**2

def windowin(arr1,p1,p2):
    return np.heaviside(arr1 - p1, 1) - np.heaviside(arr1 - p2,1)

def windowex(arr1,p1,p2):
    return np.heaviside(arr1 - p1, 0) - np.heaviside(arr1 - p2,0)

def setup_sphere(Rad=1):
    '''
    By defining everything in terms of spherical coordinates, we can generate
    displacements and shifts without regard to the underlying metric.
    Inputs:
        Rad: radius of underlying sphere
        ri: position vectors
        dr: shift amount
    Returns: 
        disp_sphere: the spherical displacement in theta and phi (which is not 
        accurate, but we can't use this to calculate distances anyway because
        JAX would default to a Euclidean distance in theta-phi space)
        shift_sphere: the shift function between points. 
    '''
    def disp_sphere(r1,r2,**unused_kwargs):
        '''
        Dummy function that gives displacement in (theta,phi) space.
        '''
        return r1 - r2

    def shift_sphere(r, dr, **unused_kwargs):
      '''
      Shift function enforcing spherical boundary conditions through the inclusion of
      Heaviside step functions. Because we avoid the use of conditionals, this shift 
      function can be compiled using jit -- which in tests has reduced compilation time
      by as much as a factor of 20.  
      '''
      #theta needs unique wrapping condition due to taking up 1/2 of 2pi space
      start = 0
      stop = len(r[:,0])

      newTheta = np.array(r[:,0] + dr[:,0]) #this produces an (N, ) array - reshape later
      newTheta = np.mod(newTheta,2*np.pi)
      newPhi = np.array(r[:,1] + dr[:,1])
      newPhi = np.mod(newPhi,2*np.pi)
    
      newPhi = (newPhi*windowin(newTheta,0,np.pi) + 
                (newPhi+np.pi)*windowex(newTheta,np.pi,10*np.pi) +
                (newPhi+np.pi)*windowex(newTheta,-8*np.pi,0))

      newTheta = (newTheta*windowin(newTheta,0,np.pi) + 
                (2*np.pi - newTheta)*windowex(newTheta,np.pi,10*np.pi) +
                (-newTheta)*windowex(newTheta,-8*np.pi,0))
    
      newTheta = np.mod(newTheta,2*np.pi)
      newPhi = np.mod(newPhi,2*np.pi)
    
      #necessary reshaping to get (N,1) and enable concatenation
      newTheta = np.reshape(newTheta,(stop,1)) 
      newPhi = np.reshape(newPhi,(stop,1))
      rnew = np.array(np.concatenate((newTheta,newPhi),axis=1))
    
      return rnew

    def sphere_dist(r1, r2, radius=Rad, **unused_kwargs):
      '''
      finds distance between two points on a sphere. default spherical radius is 1.
      r[0] = theta. r[1] = phi.
      args: 
        r1: first position (theta,phi)
        r2: second position (theta,phi)
        rad: underlying spherical radius
      returns:
        dist: a float-valued distance between two points
      '''

      #flatten for proper indexing
      r1 = r1.reshape(-1)
      r2 = r2.reshape(-1)

      #convert to longitude/latitude 
      t1 = r1[0] - np.pi/2
      t2 = r2[0] - np.pi/2

      #calculate scalar distance
      dist = Rad*2*np.arcsin(np.sqrt( hav(t1-t2) + (1 - hav(t1-t2) - hav(t1+t2))*hav(r1[1]-r2[1])))
      return dist

    return disp_sphere, exp_shift_sphere, sphere_dist
 

def soft_sphere_simulation_force(metric, energy_fn, Np,
                                sigma, alpha, epsilon, radius = 1):
  '''
  Initializes the force function for a soft sphere simulation in a way that
  circumvents JAX_MD's internal force calculation. This is necessary because, 
  at least at the time of writing this function, JAX_MD's force calculations
  on the sphere yielded NaNs across every entry. This avoids that by calculating
  gradients prior to vectorization. Piggybacks off
  of the JAX_MD definition of soft sphere energy with variables alpha,  
  epsilon, and sigma. 

  Inputs:
    metric: a function that returns the distance between two points in your 
      space.
    N:  the number of particles in the simulation. 
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
  
  #if these are defined within indiv_force, get NaNs out again
  energyGrad = jax.grad(energy_fn)
  metGrad = jax.grad(metric)

  def indiv_force(r1,r2):
    '''
    force on r1 - vector output. 
    '''
    #scales force by (1/R) in theta and (1/Rsin(theta)) in phi -- idea 
    #is to move from mapped (on sphere) back to unmapped (in abstract
    #theta-phi space) coordinates before we shift.
    distScaling = np.array([1/radius,1/(radius*np.sin(r1[0]))]) 
    unscaledForce = -energyGrad(metric(r1,r2), sigma = sigma, 
                                alpha = alpha, epsilon = epsilon)*metGrad(r1,r2)
    return distScaling*unscaledForce

  def map_function(fxn):
    return vmap(vmap(fxn,(0,None),0),(None,0),0)

  def remove_diag(matrix):
    '''
    Replicates smap's diagonal mask.
    '''
    matrix = np.nan_to_num(matrix) #change nans to zero
    mask = f32(1.0) - np.eye(N, dtype=matrix.dtype)
    if len(matrix.shape) == 3:
      mask = np.reshape(mask, (N, N, 1))
    return mask*matrix 

  def sum_rows(matrix):
    return np.einsum('ijk->jk',matrix)

  def pairwise_forces(R):
    '''
    Function that maps the individual forces to generate pairwise forces. Summing over row i
    yields the force on particle i.  
    '''
    fn_mapped = map_function(indiv_force) 
    forceMat = remove_diag(fn_mapped(R,R))
    return sum_rows(forceMat)
  
  return pairwise_forces
  
