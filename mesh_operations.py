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
  
def testMakeTriangulation(points, height, vertexList = None, rhoTolerance=1):
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
      edgeList = []
      #define cadre of height derivatives for generating curvatures/normals
      hx = grad(height,0)
      hy = grad(height,1)
      hxx = grad(grad(height,0),0)
      hyy = grad(grad(height,1),1)
      hxy = grad(grad(height,0),1)

      #Helper functions necessary for triangulation
      def ksFunction(dx,dy,dxx,dxy,dyy,**kwargs):
        #define principal curvatures/normals
        kmin = (dxx - sqrt(dxy**2 + (dxx-dyy)**2)+dyy)/(2*sqrt(1+dx**2+dy**2))
        kmax = (dxx + sqrt(dxy**2 + (dxx-dyy)**2)+dyy)/(2*sqrt(1+dx**2+dy**2))
        if kmin <= kmax: 
          return kmin, kmax
        return kmax, kmin
      
      def normalFunction(dx,dy,**kwargs):
        #This must be normalized to preserve natural slope magnitudes when 
        #determining intersects. The slopes, in turn, must *not* be normalized.
        norm = sqrt(1+dx**2 + dy**2)
        return np.array([-dx/norm,-dy/norm,1/norm])

      def neighborhoodSize(kmax,kmin,**kwargs):
        #dynamically sized neighborhood dependent on principal curvatures.
        #this function should never be called when kmax or kmin = 0

        return min(rhoTolerance,2*abs(kmax/kmin)) #must be positive to compare to E. distances

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

      appendedSets = [] #little trick to avoid extra edge appending
      for i in range(np.shape(points)[0]):
        #define current point and Euclidean distances between this point and
        #all other points
        r = points[i]
        #print('Checking point ', i)
        distances = pointToAll(euclideanDistance)(r,points) #(N,) list of distances to our source point
        distances = distances.at[i].set(1000000) #avoid ever returning self with np.where()
        numpyDistances = np.copy(distances) #create a numpy copy so we can use argpartition, which isn't available for jax arrays

        x = r[0]
        y = r[1]

        #numerical derivative values for curvatures, normal, neighborhood
        dx = hx(x,y)
        dy = hy(x,y)
        dxx = hxx(x,y)
        dxy = hxy(x,y)
        dyy = hyy(x,y)


        #numerical curvatures, normal, neighborhood
        kmin,kmax = ksFunction(dx,dy,dxx,dxy,dyy)

        localNormal = normalFunction(dx,dy)
        kNeighborhood = neighborhoodSize(kmax,kmin)
        #print('Neighborhood size is: ', 2*kmax/kmin)

        #average number of nearest neighbors should be ~6 for a decently large
        #triangulation w/poisson sampled points (implement!). So if we check a few
        #points than that, we should be checking approx. all relevant points.
        #Should really only run the risk of being a little slow for small data
        #sets, or miss a neighbor or two in larger ones
        if kmax*kmin == 0:
          avgNeighborNumber = 11 #this is a test value! This whole part of the function is a test value!
          neighborIndices = jnp.copy(np.argpartition(numpyDistances,avgNeighborNumber))[:avgNeighborNumber]
        else: 
          neighborIndices = jnp.where(distances < kNeighborhood)[0]
          #print('Number of candidate neighbors: ', len(neighborIndices))

        neighborPoints = points[neighborIndices]
        neighborVectors = neighborPoints - r 

        normalOverlap = jnp.dot(neighborVectors,localNormal) #(N,) scalar dot products

        #neighbor vectors come out as (1,N,d) shaped arrays, which we acknowledge in indices later
        projectedNeighborVectors = neighborVectors - jnp.einsum('i,j->ij',jnp.dot(neighborVectors,localNormal),localNormal)
        projectedNeighborPoints = r + projectedNeighborVectors 
        projectedNeighborAngles = jnp.arctan2(projectedNeighborVectors[:,1],projectedNeighborVectors[:,0])

        #sort points by lowest to highest angle (angles range [-pi,pi])
        indexOrder = projectedNeighborAngles.argsort()

        sortedNeighbors = neighborPoints[indexOrder]
        sortedProjectedNeighbors = projectedNeighborPoints[indexOrder]
        sortedNeighborVectors = projectedNeighborVectors[indexOrder]
        sortedNeighborDistances = np.linalg.norm(sortedNeighborVectors,axis=1)
        sortedIndices = neighborIndices[indexOrder]
        #sorted indices necessary to draw edges
        
        #for j in range(len(neighborIndices)):
          #neighborIndices = neighborIndices.at[j].set(neighborIndices[indexOrder[j]])
        
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

          elif (not isNeighbor): #this else CANNOT happen for the first vertex as a rule
            verticesToTest.remove(verticesToTest[bIndex])
            bIndex -= 1

      return edgeList


'''
Segments of code used for sampling, with density going as the inverse of curvature. 
'''

def ksFunction(dx,dy,dxx,dxy,dyy,**kwargs):
        kmin = (dxx - sqrt(dxy**2 + (dxx-dyy)**2)+dyy)/(2*sqrt(1+dx**2+dy**2))
        kmax = (dxx + sqrt(dxy**2 + (dxx-dyy)**2)+dyy)/(2*sqrt(1+dx**2+dy**2))
        if kmin <= kmax: #should always happen!
          return kmin, kmax
        return kmax, kmin

def inverseCurvatureSampling(height,ywidth,xwidth, Ndef):
  '''
  Samples a height map height(x,y) such that points are
  closer together in more curved regions than they are 
  in less curved/flat regions. 

  With how long it takes right now, it could easily be 
  called ``Intense'' curvature sampling...
  '''
  hx = grad(height,0)
  hy = grad(height,1)
  hxx = grad(grad(height,0),0)
  hyy = grad(grad(height,1),1)
  hxy = grad(grad(height,0),1)

  #two lists to store x and y values. 
  x, y = 0.0,0.0 
  xs, ys, zs = [], [], []

  #what the spacing will be if it's flat (a grid!)
  xSpacingDefault = xwidth/Ndef
  ySpacingDefault = ywidth/Ndef

  toleranceX, toleranceY = xSpacingDefault, ySpacingDefault

  curvatureDeScaling = 4 #meant to tighten sampling so that we tend to 
                         #oversample rather than undersample, there's probably 
                         #a better precise number here

  #So. We "need" to do a while loop because the lengths of x and y are not 
  #guaranteed. Unfortunately, we can only calculate curvatures along
  #the principal directions, which are, for surfaces curved in both 
  #x and y, neither x or y. My dumb version of the solution here is 
  #to use only the biggest curvature in defining my point spacing. 
  finishedLoops = 0
  while x-xwidth < toleranceX:
    if finishedLoops % 5 == 0:
      print('Loops completed = ', finishedLoops)
    y = 0.0 #reset y to zero when we move in x. 
    dx = hx(x,y)
    dy = hy(x,y)
    dxx = hxx(x,y)
    dxy = hxy(x,y)
    dyy = hyy(x,y)
    #numerical curvatures kmin and kmax -- principal directions not neceesarily x and y 
    kmin,kmax = ksFunction(dx,dy,dxx,dxy,dyy) #calculating kmin is unnecessary

    if kmax == 0: #avoid explosions
      x += xSpacingDefault
    else:
      x += min(xSpacingDefault,abs(xSpacingDefault/(curvatureDeScaling*kmax))) #curvatures can be negative!

    while y-ywidth < toleranceY:
      #always add to array together to avoid misshappen combinations
      xs.append(x)
      ys.append(y)
      zs.append(height(x,y))

      dx = hx(x,y)
      dy = hy(x,y)
      dxx = hxx(x,y)
      dxy = hxy(x,y)
      dyy = hyy(x,y)
      #numerical curvatures kmin and kmax -- principal directions not necessarily x and y 
      kmin,kmax = ksFunction(dx,dy,dxx,dxy,dyy) #calculating kmin is unnecessary

      if kmax == 0: #avoid explosions
        y += ySpacingDefault
      else: 
        y += min(ySpacingDefault,abs(ySpacingDefault/(curvatureDeScaling*kmax)))

    finishedLoops += 1

  return jnp.transpose(jnp.array([xs,ys,zs]))


'''
Segments of code used to prune overlapping edges. 
'''


def findSlope(point1,point2):
        #find the slope of a segment
        slope = (point2-point1)
        
        #make sure slope is left-to-right in x. Points are only considered in 
        #left-right order in segmentIntersect to ensure consistency. 
        if slope[0] < 0:
          slope *= -1
        
        return slope #no normalization to retain segment length

def segmentIntersect(pointa1,pointa2,pointb1,pointb2):
        #determine which vertex is leftward (in x) for each pair so we always 
        #know we're walking forward along the segment. Slopes, via slope finding
        #function, are guaranteed to be negative-to-positive in x. 
        xa1 = pointa1[0]
        xa2 = pointa2[0]
        xb1 = pointb1[0]
        xb2 = pointb2[0]
        if xa2 < xa1: 
          store = pointa1 
          pointa1 = pointa2 #point 1 should always be the leftmost point in x
          pointa2 = store

        if xb2 < xb1: 
          store = pointb1 #point 1 should always be the leftmost point in x
          pointb1 = pointb2
          pointb2 = store

        #get components for component wise formulae -- assume 3d
        slopea = findSlope(pointa1,pointa2)
        slopeb= findSlope(pointb1,pointb2)
        slopeaNorm = slopea/np.linalg.norm(slopea)
        slopebNorm = slopeb/np.linalg.norm(slopeb)

        sx1 = slopea[0]
        sy1 = slopea[1]
        sz1 = slopea[2]
        
        xa = pointa1[0]
        ya = pointa1[1]
        za = pointa1[2]

        sx2 = slopeb[0]
        sy2 = slopeb[1]
        sz2 = slopeb[2]

        xb = pointb1[0]
        yb = pointb1[1]
        zb = pointb1[2]

        if (slopeaNorm==slopebNorm).all()  and ((pointa1==pointb1).all() or (pointa2 == pointb2).all()):
          #colinearity check -- because we've organized the points left to 
          #right, we know the the indices of the points will line up somewhere 
          #for colinear segments

          #this may fail due to numerical precision!
          return True #technically there's an intersection + we need to delete longer edge

        #alpha and beta are solutions of 2 eq. system of linear eqs:
        # sy1*alpha + y01 = sy2*beta + y02
        # sx1*alpha + x01 = sx2*beta + x02
        #(underscore stands for the specific coordinate value)
        if (sy2*sx1-sy1*sx2) == 0:
          #print('Parallel.') #Colinearity?
          return False

        alpha = (sx2*ya - sx2*yb - sy2*xa + sy2*xb)/(sy2*sx1-sy1*sx2)
        beta = (sx1*ya-sx1*yb - sy1*xa+sy1*xb)/(sy2*sx1-sy1*sx2)
        

        if np.isclose(sx1*alpha + xa, sx2*beta + xb,atol=1e-8):
          #print('Intersection!')    
          if np.sign(alpha) == np.sign(beta):
            if (alpha < 1) and (alpha > 0) and (beta <1) and (beta > 0):
              return True
        
        return False

def removeExcessEdges(edges,positions,vertexList = None):
  runningEdges = edges 
  #all pairs of edges -- arrays can index other arrays
  edgePositions = [np.array(list(i.getVerts())) for i in edges] 
  edgeLengths = [i.getLength() for i in edges]
  positionPairs = np.array([positions[i] for i in edgePositions]) #all pairs of positions - N by 2 by d 

  #pruning loop - check all pairs of edges, dynamically removing from the running list
  it1 = 0 #first iterator
  it2 = 0 #second iterator
  justDeleted = False
  while it1 < len(runningEdges):
    if justDeleted:
      #repeat of above code when the edge list changes
      edgePositions = [np.array(list(i.getVerts())) for i in edges]     
      edgeLengths = [i.getLength() for i in runningEdges]
      positionPairs = np.array([positions[i] for i in edgePositions]) 
      justDeleted = False

    theyIntersect= segmentIntersect(positionPairs[it1,0],positionPairs[it1,1],
                                    positionPairs[it2,0],positionPairs[it2,1])
    if theyIntersect: 
      if it1==it2:
        if it2 >= len(runningEdges) - 1:
          it2 = 0 
          it1 +=1 
          continue
        else: 
          it2 += 1
          continue

      #default to remove longer edge, just for fun..
      if edgeLengths[it1] > edgeLengths[it2]:
        #print('Deleting edge ', edgePositions[it1])
        justDeleted = True
        pruningEdge = runningEdges[it1]
        del runningEdges[it1]
        if vertexList != None:
           edgeVertices = np.array(list(pruningEdge.getVerts()))
           vertexList[edgeVertices[0]].removeConnection(pruningEdge)
           vertexList[edgeVertices[1]].removeConnection(pruningEdge)

      else:
        #print('Deleting edge ', edgePositions[it2])
        justDeleted = True
        pruningEdge = runningEdges[it2]
        del runningEdges[it2]
        if vertexList != None:
           edgeVertices = np.array(list(pruningEdge.getVerts()))
           vertexList[edgeVertices[0]].removeConnection(pruningEdge)
           vertexList[edgeVertices[1]].removeConnection(pruningEdge)



    if it2 >= len(runningEdges) - 1:
      it1 += 1
      it2 = 0 
    else: 
      it2 += 1

  return runningEdges
