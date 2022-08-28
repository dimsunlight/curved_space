import numpy as np
import random as random

import jax
from jax.config import config ; config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import jit
from jax import lax
from jax import vmap
from jax import grad
from jax.numpy import sqrt,cos, sin
from mesh_objects import Vertex, Edge, HeightMeshGraph, Face

def euclideanDistance(r1,r2,**kwargs):
      #using jnp for later vectorization
      return jnp.linalg.norm(r1 - r2)

def pointToAll(function, **kwargs):
      return vmap(function,(None,0),0)
      
def makeTriangulation(points, height, vertexList=None,rhoTolerance=1):
          '''
          Will eventually be: create a Delaunay triangulation to serve as the
          mesh by drawing edges (with Euclidean distances) between appropriate 
          points. Happens every initialization of MeshGraph, so we'd prefer not
          to initialize too often -- Delaunay triangulations take at least
          NlogN time, with worst case N^2. 

          algorithm based on Gopi et al., Computer Graphics Forum 2000. Simplified
          greatly by the fact that we automatically know normals and principal 
          curvatures from the definition of a Monge patch, removing several steps of
          approximation. Derivatives are calculated with Jax autodifferentiation 
          utilities. 
          '''
          #define cadre of height derivatives for generating curvatures/normals.
          #for now, leaving these to be defined within triangulation maker -- 
          #could make them exist across the heightmeshgraph class
          hx = grad(height,0)
          hy = grad(height,1)
          hxx = grad(grad(height,0),0)
          hyy = grad(grad(height,1),1)
          hxy = grad(grad(height,0),1)

          '''
          Helper functions for triangulation
          '''
          def calculateKs(dx,dy,dxx,dxy,dyy,**kwargs):
            #define principal curvatures/normals
            kmin = (dxx - sqrt(dxy**2 + (dxx-dyy)**2)+dyy)/(2*sqrt(1+dx**2+dy**2))
            kmax = (dxx + sqrt(dxy**2 + (dxx-dyy)**2)+dyy)/(2*sqrt(1+dx**2+dy**2))
            #always ordered smallest (magnitude) to largest
            if abs(kmin) <= abs(kmax): 
              return kmin, kmax
            return kmax, kmin
          
          def normalFunction(dx,dy,**kwargs):
            #This must be normalized to preserve natural slope magnitudes when 
            #determining intersects. The slopes, in turn, must *not* be normalized.
            norm = sqrt(1+dx**2 + dy**2)
            return (1/norm)*np.array([-dx,-dy,1])

          def neighborhoodSize(kmin,kmax,nearestDist,**kwargs):
            #dynamically sized neighborhood dependent on principal curvatures.
            #this function should never be called when kmax or kmin = 0
            return min(rhoTolerance,2*nearestDist*abs(kmax/kmin)) #must be positive

          def heightMax(kmax,kmin,**kwargs):
            #maximum height in the tangent plane -- said otherwise, maximum allowed 
            #z value above current point 
            return abs((sqrt(1+4*(kmax/kmin)**2)-1)/kmin)

          def projection(dr,normal,**kwargs):
            '''
            Simple function to project a point/vector in xyz into the tangent
            plane of the surface near a given point. 
            '''
            #dr is (m,d), where m is number of neighbor points, and normal is (d,) 
            normaldotdr = np.einsum('ij,j->i',dr,normal) #dimensions just (N,)
            return dr - np.einsum('i,j->ij',normaldotdr,normal) #(N,) with (d,) to (N,d)

          def bisectorSlope(point1,point2,normal):
            #finds point on perpendicular bisector of the line between 3D points 1 and 2
            pointSlope = (point2 - point1)
            orthogonalSlope = np.cross(pointSlope,normal) #(3,) vector slope in-plane
            midpoint = (point1+point2)/2
            return orthogonalSlope,midpoint #testing for easier to read slopes
            #return orthogonalSlope/np.linalg.norm(orthogonalSlope),midpoint

          def intersectionPoint(slope1,slope2,midpoint1,midpoint2):
            #get components for component wise formulae
            sx1 = slope1[0]
            sy1 = slope1[1]
            sz1 = slope1[2]
            
            x01 = midpoint1[0]
            y01 = midpoint1[1]
            z01 = midpoint1[2]

            sx2 = slope2[0]
            sy2 = slope2[1]
            sz2 = slope2[2]

            x02 = midpoint2[0]
            y02 = midpoint2[1]
            z02 = midpoint2[2]

            #alpha and beta are solutions of 2 eq. system of linear eqs:
            # sy1*alpha + y01 = sy2*beta + y02
            # sx1*alpha + x01 = sx2*beta + x02
            #(underscore stands for the specific coordinate value)

            alpha = (sx2*y01 - sx2*y02 - sy2*x01 + sy2*x02)/(sy2*sx1-sy1*sx2)
            beta = (sx1*y01-sx1*y02 - sy1*x01+sy1*x02)/(sy2*sx1-sy1*sx2)

            if (sy2*sx1-sy1*sx2) == 0:
              #print('Parallel.')
              return np.array([None])

            elif np.isclose(sx1*alpha + x01, sx2*beta + x02,atol=1e-5):
              #print('Intersection!')
              return np.array([alpha*sx1+x01, alpha*sy1+y01, alpha*sz1+z01])
            
            return np.array([None])

          def isLeft(a,b,c):
            '''
            from https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
            Returns true (point is on the left) or false (point is not). a is on the 
            left side of the dividing line, b is on the right side. Technically
            the ==0 case implies colinearity, but I need to give it true or false 
            so it doesn't stop the code. a and b are both points on the line, c is 
            the test point. Accurately identifying true or false isn't important, points on 
            the same side just have to match (so precision, not accuracy).
            '''
            test = ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]))
            
            return test > 0

          def isDelaunayNeighbor(r,pointB,pointA,pointC,normal):
            #checks if p1 is a delaunay neighbor of r,
            #assuming point 1 is the point between points 2 and 3
            #coefficients of the perpendicular bisectors of r - p_i
            slopeB,midpointB = bisectorSlope(r,pointB,normal)
            slopeA,midpointA = bisectorSlope(r,pointA,normal) #-1 index gives last element
            slopeC,midpointC = bisectorSlope(r,pointC,normal)

            '''
            Following is not really understandable without looking at Gopi et al. 
            2000. Basically, we are drawing the Voronoi edges and checking to see
            if the midpoint of the line between point i and our test point 
            falls within them along with the test point. If it does, then point i
            is a Delaunay neighbor. 
            '''

            #define intersection point of perpendicular bisectors A and C (matching
            #Gopi et al. notation), other algorithm quantities 
            IAC = intersectionPoint(slopeA,slopeC,midpointA,midpointC)
            #check for parallel/no intersections
            if (len(IAC) == 1):
              if IAC[0] == None:
                #Parallel or no intersection.
                return False

            slopeIB, intersectIB = slopeB, IAC #define LI1 as line parallel to bisector of line r1 passing through I23
            linepoint1 = slopeIB*(-1)+intersectIB
            linepoint2 = slopeIB*(1)+intersectIB

            rside = isLeft(linepoint1,linepoint2,r) #returns TRUE/FALSE
            mpBside = isLeft(linepoint1,linepoint2,midpointB) #returns TRUE/FALSE

            return rside == mpBside #if same side, they are D. neighbors 



          #Primary Loop!
          appendedSets = [] #little trick to avoid
                            #extra edge appending across loops
          edgeList = []
          xVector = np.array([1,0,0])
          #testLoopCount = -1


          for i in range(np.shape(points)[0]):
            #define current point and Euclidean distances between this point and
            #all other points
            r = points[i]
            distances = pointToAll(euclideanDistance)(r,points) #(N,) list of distances to our source point
            distances = distances.at[i].set(1000000) #avoid ever returning self with np.where()
            numpyDistances = np.copy(distances) #create a numpy copy so we can use argpartition, 
                                                #which isn't available for jax arrays
            s = min(distances) #distance to nearest neighbor
            x = r[0]
            y = r[1]

            #numerical derivative values for curvatures, normal, neighborhood
            dx = hx(x,y)
            dy = hy(x,y)
            dxx = hxx(x,y)
            dxy = hxy(x,y)
            dyy = hyy(x,y)


            #numerical curvatures, normal, neighborhood
            kmin,kmax = calculateKs(dx,dy,dxx,dxy,dyy)
            localNormal = normalFunction(dx,dy)
            kNeighborhood = neighborhoodSize(kmin,kmax,s)
            zmax = heightMax(kmax,kmin)

            #average number of nearest neighbors should be ~6 for a decently large
            #triangulation w/poisson sampled points (implement!). So if we check a few
            #points than that, we should be checking approx. all relevant points.
            if kmax*kmin == 0:
              avgNeighborNumber = 10 #this is a test value!
              neighborIndices = jnp.copy(np.argpartition(numpyDistances,avgNeighborNumber))[:avgNeighborNumber]
            else: 
              neighborIndices = jnp.where(distances < kNeighborhood)[0]

            neighborPoints = points[neighborIndices]
            neighborVectors = neighborPoints - r 

            normalOverlap = jnp.dot(neighborVectors,localNormal) #(N,) scalar dot products

            #neighbor vectors come out as (1,N,d) shaped arrays, which we acknowledge in indices later
            projectedNeighborVectors = projection(neighborVectors,localNormal) #project into tangent plane
            projectedNeighborPoints = r + projectedNeighborVectors 

            '''
            here; remove everything with projected height greater than z max.
            At the moment, this does absolutely nothing.  
            '''
            tooHigh = jnp.array([jnp.where(abs(projectedNeighborVectors[:,2]) > zmax)])
            neighborPoints = jnp.delete(neighborPoints, tooHigh,axis=0)
            neighborVectors = jnp.delete(neighborVectors, tooHigh,axis=0)
            projectedNeighborPoints = jnp.delete(projectedNeighborPoints, tooHigh, axis=0)
            projectedNeighborVectors= jnp.delete(projectedNeighborVectors, tooHigh, axis=0)

            #to get angles -- we need to take arctan2 of each vector with an arbitrary
            #straight `x' vector in the plane. 
            # so arctan2(cross(vectors[i],xvector),dot(vectors[i],xvector)
            
            tangentXVector = (xVector - np.dot(xVector,localNormal)*localNormal) - r
            tangentXArray = jnp.zeros_like(projectedNeighborVectors) + tangentXVector
            normalArray = jnp.zeros_like(projectedNeighborVectors) + localNormal

            crosses = np.cross(tangentXArray,projectedNeighborVectors)

            #retain signed angle from "x" axis
            dotSigns = np.sign(np.einsum('ij,ij->i',normalArray,crosses))  
            crossMagsWithSign = np.einsum('i,j->i',dotSigns,np.linalg.norm(crosses,axis=1))
            dots = np.einsum('ij,ij->i',projectedNeighborVectors,tangentXArray)
            projectedNeighborAngles = jnp.arctan2(crossMagsWithSign,dots)
            

            #sort points by lowest to highest angle (angles range [-pi,pi])
            indexOrder = projectedNeighborAngles.argsort()

            #Good lord this is a lot of list storage -- they're all tiny in principle, but 
            #it really makes me feel like I could get away with doing less...
            projectedNeighborAngles = projectedNeighborAngles[indexOrder]
            sortedNeighbors = neighborPoints[indexOrder]
            sortedProjectedNeighbors = projectedNeighborPoints[indexOrder]
            sortedNeighborVectors = projectedNeighborVectors[indexOrder]
            sortedNeighborDistances = np.linalg.norm(sortedNeighborVectors,axis=1)
            sortedIndices = neighborIndices[indexOrder]
            
            #now find Delaunay neighbors. 
            verticesToTest = list(range(jnp.shape(sortedProjectedNeighbors)[0]))

            #loop initializers
            bIndex = np.argmin(sortedNeighborDistances) #closest point, guaranteed to be a neighbor
            startingB = verticesToTest[bIndex]
            countedSets = []

            while len(verticesToTest) > 2: #could this be leaving vertices untested because we remove others from the queue?

              bIndex = bIndex % len(verticesToTest)
              aIndex = bIndex - 1
              cIndex = (bIndex + 1) % len(verticesToTest) #can be out of bounds

              A = verticesToTest[aIndex]
              B = verticesToTest[bIndex]
              C = verticesToTest[cIndex]

              pointA = sortedProjectedNeighbors[A]
              pointB = sortedProjectedNeighbors[B]
              pointC = sortedProjectedNeighbors[C]
              
              isNeighbor = isDelaunayNeighbor(r,pointB,pointA,pointC,localNormal)
              if(isNeighbor):
                #check if we've visited this node before
                if set(verticesToTest).issubset(set(countedSets)): #true if counted sets has all elements of the vertices to test
                  #print('Checked all possibilities for candidate ', i)
                  break

                countedSets.append(verticesToTest[bIndex])
                edgeLength = euclideanDistance(r,sortedNeighbors[B])
                currentVertex = sortedIndices[verticesToTest[bIndex]].item() #.item() prevents an array from being returned
                pointSet = {i,currentVertex}
                if pointSet != {i}: #this should basically never happen unless you use too few points
                  if pointSet not in appendedSets:
                    drawEdge = Edge(pointSet,edgeLength)
                    edgeList.append(drawEdge)
                    if vertexList != None:
                      vertexList[i].addConnection(drawEdge)
                      vertexList[currentVertex].addConnection(drawEdge)
                    appendedSets.append(pointSet)

                bIndex += 1

              elif (not isNeighbor): #this elif CANNOT happen for the first vertex as a rule
                verticesToTest.remove(verticesToTest[bIndex])
                bIndex -= 1

          return edgeList, vertexList

def removeExcessEdges(edges,positions,vertexList = None):
  #pruning procedure to delete edges that overlap in the xy plane of our 
  #height function based mesh
  def twoDFindSlope(point1,point2):
          #find the slope of a segment
          slope = (point2-point1)
          slopexy = jnp.array([slope[0],slope[1]])
          #make sure slope is left-to-right in x. Points are only considered in 
          #left-right order in segmentIntersect to ensure consistency. 
          if slopexy[1] < 0:
            slopexy *= -1 
          if slopexy[0] < 0:
            slopexy *= -1
          
          return slopexy #no normalization to retain segment length

  def twoDSegmentIntersect(pointa1,pointa2,pointb1,pointb2):
          '''
          Determines whether two segments intersect, but only in 
          the two-dimensional xy plane. 
          '''

          #determine which vertex is leftward (in x) for each pair so we always 
          #know we're walking forward along the segment. Slopes, via slope finding
          #function, are guaranteed to be negative-to-positive in x or, if 
          #x does not change, negative-to-positive in y.
          xa1 = pointa1[0]
          xa2 = pointa2[0]
          xb1 = pointb1[0]
          xb2 = pointb2[0]
          ya1 = pointa1[1]
          ya2 = pointa2[1]
          yb1 = pointb1[1]
          yb2 = pointb2[1]
          #point 1 should always be the leftmost point in x or lower point in y
          if xa1 == xa2:
            if ya2 < ya1: #reverse point order 
              pointa1,pointa2 = pointa2,pointa1
          
          if xa1 != xa2:
            if xa2 < xa1: #reverse point order
              pointa1,pointa2 = pointa2,pointa1

          if xb1 == xb2:
            if yb2 < yb1: #reverse point order
              pointb1,pointb2 = pointb2,pointb1

          if xb1 != xb2:
            if xb2 < xb1: #reverse point order
              pointb1,pointb2 = pointb2,pointb1

          #get components for component wise formulae -- assume 3d
          slopea = twoDFindSlope(pointa1,pointa2)
          slopeb= twoDFindSlope(pointb1,pointb2)

          slopeaNorm = slopea/np.linalg.norm(slopea)
          slopebNorm = slopeb/np.linalg.norm(slopeb)

          sx1 = slopea[0]
          sy1 = slopea[1]
          sx2 = slopeb[0]
          sy2 = slopeb[1]

          xa1 = pointa1[0]
          ya1 = pointa1[1]

          xb1 = pointb1[0]
          yb1 = pointb1[1]
          
          #parallel and collinearity checker
          if np.allclose(slopeaNorm,slopebNorm):
            #if this is true, parallel or colinear - now determine if collinear!
            #first: do we start or end at the same place? order guaranteed
            if (pointa1 == pointb1).all() or (pointa2 == pointb2).all():
              return True

            #most of the finagling here involves dealing with the zero slope case.
            alpha1 = (xb1-xa1)/sx1 #NaN if vertical line
            alpha2 = (yb1-ya1)/sy1 #NaN if horizontal line

            #check for slopes zero, then check if start points or endpoints match
            if (sx1 == 0) or (sy1 == 0):
              #fixing NaNs in slope bc 0 slope -> infinite arc length. 
              #Find overlaps through alpha bounds -- i.e., 0 <= alpha < 1 condition

              if (sx1 == 0) and (xa1 == xb1): #if we're on the same vertical line
                alpha1 = alpha2 #collinearity only depends on y -- x holds on vertical

              elif (sy1 == 0) and (ya1 == yb1): #if we're on the same horizontal line
                alpha2 = alpha1 #collinearity only depends on x -- y holds on horizontal
                
            if np.isclose(alpha1,alpha2) and (alpha1 < 1) and (alpha1 >= 0):
              #returns true if segments are not completely collinear but overlap
              return True
            return False #should only happen when lines are parallel

          #denominators are zero if parallel or collinear -- should never get
          #divide by zero because of above check
          alpha = (sx2*ya1 - sx2*yb1 - sy2*xa1 + sy2*xb1)/(sy2*sx1-sy1*sx2)
          beta = (sx1*ya1-sx1*yb1 - sy1*xa1+sy1*xb1)/(sy2*sx1-sy1*sx2)
          
          if np.isclose(sx1*alpha + xa1, sx2*beta + xb1,atol=1e-8):
            if np.sign(alpha) == np.sign(beta):
              #using .9999 to avoid roundoff errors on quantities that should be 1
              if (alpha < .9999) and (alpha > 0) and (beta < .9999) and (beta > 0):
                return True
          
          return False




  runningEdges = edges #copy of edge list 
  runningVertices = vertexList #copy of vertex list

  #all pairs of edges -- arrays can index other arrays
  edgePositions = [np.array(list(edge.getVerts())) for edge in edges] 
  edgeLengths = [edge.getLength() for edge in edges]
  positionPairs = np.array([positions[pair] for pair in edgePositions]) 

  #check all pairs of edges, dynamically removing from the running list
  it1 = 0 #first iterator
  it2 = 0 #second iterator
  newit1 = 0 #iterator placeholders
  newit2 = 0

  printEvery = 5
  justDeleted = False
  printed = False
  print('Beginning pruning loop.')
  #this is basically a nested for loop with dynamic list size.
  while it1 < len(runningEdges):
    if (it1 % printEvery ==0) and (printed == False):
      print('On pruning loop iteration ', it1)
      print('Current running edge length: ', len(runningEdges))
      printed = True

    if justDeleted:
      #repeat of above code when the edge list changes
      edgePositions = [np.array(list(i.getVerts())) for i in runningEdges]     
      edgeLengths = [i.getLength() for i in runningEdges]
      positionPairs = np.array([positions[i] for i in edgePositions]) 
      justDeleted = False

    theyIntersect= twoDSegmentIntersect(positionPairs[it1,0],positionPairs[it1,1],
                                    positionPairs[it2,0],positionPairs[it2,1])
    if theyIntersect: 
      if it1==it2:
        if it2 >= len(runningEdges) - 1:
          newit1 = it1 +1
          newit2 = 0 
          it1 +=1 
          
        elif it2 < len(runningEdges) -1: 
          newit1 = it1
          newit2 = it2+1
          
        it1 = newit1
        it2 = newit2
        continue
      '''
      #debugging statement
      print('Intersection!')
      print('Points ', positionPairs[it1,0], positionPairs[it1,1], ' and ', positionPairs[it2,0],positionPairs[it2,1])
      print('Indices ', edgePositions[it1], edgePositions[it2])
      '''
      #default to remove longer edge, just for fun..
      if edgeLengths[it1] >= edgeLengths[it2]:
        #if we delete the it1 edge, we just move to evaluating all matches for
        #the it1+1 edge, no harm no foul
        justDeleted = True
        pruningEdge = runningEdges[it1]
        del runningEdges[it1]
        if vertexList != None:
           edgeVertices = np.array(list(pruningEdge.getVerts()))
           runningVertices[edgeVertices[0]].removeConnection(pruningEdge)
           runningVertices[edgeVertices[1]].removeConnection(pruningEdge)
        it2 = 0
        continue 

      elif edgeLengths[it1] <= edgeLengths[it2]:
        #if we delete the it2 edge, we need to check and see if we raised it1 
        #to a later edge on accident -- namely, if we deleted something before
        #it1's edge or after. it1 never equals it2. Also avoid skips in it2 via
        #subtraction
        justDeleted = True
        pruningEdge = runningEdges[it2]
        del runningEdges[it2]
        if vertexList != None:
           edgeVertices = np.array(list(pruningEdge.getVerts()))
           runningVertices[edgeVertices[0]].removeConnection(pruningEdge)
           runningVertices[edgeVertices[1]].removeConnection(pruningEdge)
        if it2 < it1:
          it1 -= 1
        it2 -= 1
        continue 
    
    if it2 >= len(runningEdges) - 1: #if we are at or above our maximum index
      printed = False
      newit1 = it1+1
      newit2 = 0 

    elif it2 < len(runningEdges) - 1 : 
      newit1 = it1
      newit2 = it2 + 1

    it1 = newit1
    it2 = newit2

  return runningEdges, runningVertices
