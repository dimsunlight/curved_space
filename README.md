# curved_space

OUT OF USE -- This was trial code for understanding and attempting to write mesh generation & simple curved-space simulations in Python. While it represents a fair amount of time
on my part, I never got the home-built meshing algorithm implementation to work properly; moreover, it is both faster to run and more achievable over the timescale of my PhD to do this work 
in C++ with the CGAL library. I leave this repository public (with the most recent to-do state) purely as a memory of prior work. We discarded this work only after having delved back into the literature
on 1) the mathematical structure of simulation steps and 2) the options available to do the simulations we wanted to do. Still, autograd-type approaches to distance calculations/derivative calculations
are probably interesting sources of speed-up. 

-----------------------------------------------------------------------------------------

Repository of python code for simulations in curved space. The ultimate goal is to be able to do efficient (that is, runnable on grad student time scales...) MD simulations with many particles interacting according to their geodesic distance on the target surface, which for now I assume to be an arbitrary surface described by a Monge gauge. 

Currently, mesh_objects stores the older versions of the objects relevant to a triangular mesh -- that is, the vertices, edges, faces, and total mesh itself. There are functions within those classes to generate a mesh -- namely, a list of Edges. However, the functions there are old, and nothing is actually working yet. 

More functional, but still not finished, code is written in mesh_operations. That has the newest, though still-in-testing, version of the triangulator (testMakeTriangulation) and associated helper functions, as well as code to generate points dispersed according to surface curvature (ksFunction and inverseCurvatureSampling), and code to delete excess edges (findSlope, segmentIntersect, and removeExcessEdges). None of these are quite done and dusted yet, and, once they are, I should probably sort them into their own files for easier parsing instead of storing them in one big mesh_operations files. 

TODO (in order; each step has implicit debugging): 
-Fully repair and comprehend spherical case simulations

-Evaluate current triangulator and either create an accurate pruning algorithm to remove spurious edges or rewrite the triangulator to prevent spurious edges from ever being created in the first place. We want to make this work first, even if it's slow!

-Decide whether this triangulator is really the best choice or if I should move to a different strategy -- could be a lot of rewriting

-Create Fast Marching implementation once we're done writing the mesh into existence. When this works, we can start looking at implementations of larger algorithms.

-Implement GTU if we can be certain that derivatives of the distances given by the algorithm are well-conditioned (i.e., errors are consistent and signs of derivatives are correct). 

  ->sub task for GTU -- implement point-to-all algorithm for pre-processing. This can be heat method, fast marching, improved chen and han...
  
-Run first MD simulations, buy a cake. 
