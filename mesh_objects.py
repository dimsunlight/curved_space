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

class Vertex:
    '''
    Node of the polygonal mesh. 
    Properties:
      index: (numerical) vertex label for easy recall later.
      edges: list of edges this vertex is a part of -- equivalent to connection list
      coordinates: coordinates of the point in arbitrary space (usually R3).
    '''
    def __init__(self, index, coordinates, edges = None):
        #edges = list of edges this vertex is a part of
        if edges == None:
          edges = []
        else:
          edges = [edges] #using an array would bring down typing issues
        self.index = index 
        self.edges = edges #list of vertices we're nearest neigbors to
        self.coordinates = coordinates #array of position info

    def getIndex(self):
        return self.index

    def point(self):
        return self.coordinates

    def addConnection(self,newEdge):
        self.edges.append(newEdge) #creating a new Edge would 
                                   #be wrong -- would not match
                                   #Edges outside of this 
                                   #specific list
    def removeConnection(self,deleteEdge):
        self.edges.remove(deleteEdge)

    def getConnections(self):
        return self.edges 

class Edge: 
    def __init__(self,vertexIndices,length = 1,connectedFaces = []):
      '''
      This Edge structure stores vertices just by their indices. 
      The vertex objects themselves can then be looked up within 
      the Graph data structure. The Edge itself is the connection 
      of two vertices. 
      Properties:
        vertices: (2,) set of numerical indices for connected vertices.
        length: edge length. Defaults to uniform weights (1), but we can 
          input true length within Graph.
        faces: the faces on either side of this Edge. In principle,
          this should never be empty -- but it defaults to empty because in 
          many cases storing the data is unnecessary. 
      '''

      self.vertices = vertexIndices #this should be input as a set.
      self.length = length
      self.faces = connectedFaces

    def getLength(self):
      return self.length
    def getVerts(self):
      return self.vertices
    def updateLength(self,newLength):
      self.length = newLength

        
class HeightMeshGraph:
    '''
    Super-structure creating and storing all key elements of an R2 mesh embedded
    in R3: Vertices, Edges, and Faces. Vertices are created from a list of 
    positions where list indices match vertex labels. Edges are created between
    vertices of appropriate labels. Finally, Faces can be created between 
    appropriate edges. The MeshGraph holds many of the key creation behaviors
    of the underlying data structures. 
    
    for example, MeshGraph has access to 
    Vertex coordinates, whereas Edges do not. Thus, MeshGraph can actually tell
    an Edge how long it is. <---can we not give edges actual Vertices to reference now?

    Properties:
      coordinateArray = list of coordinates for use during triangulation
      vertexList = list of vertices contained in the graph.
      edgeList = list of edges.
      faceList = list of faces.
      height = function that returns height given x and y inputs (height = 
                h(x,y,**kwargs)). Should be defined in terms of jax.numpy for 
                later derivatives.
                    
    '''
    def __init__(self,coordinateArray = None, 
                 heightFunction = None, edges=[], faces = []):
        if coordinateArray == None:
            coordinateArray = jnp.array([])
        if heightFunction == None:
            heightFunction = self.defaultHeight

        self.coordinateArray = coordinateArray #array
        self.vertexList = self.makeVertexList(coordinateArray)  
        self.edgeList = self.makeEdges()
        self.faceList = faces
        self.height = heightFunction

    def euclideanDistance(self,r1,r2,**kwargs):
      #using jnp for later vectorization
      return jnp.linalg.norm(r1 - r2)

    def pointToAllDistance(self,fxn,**kwargs):
      '''
      Generates a list of pairwise distances between a (3,) position
      array and the (N,3) list of positions. 
      '''
      return vmap(self.euclideanDistance,(None,0),0)

    def defaultHeight(x,y,**kwargs):
      '''
      written in terms of x and y so we can take automatic derivatives w.r.t. 
      both x and y.   
      '''
      return 0 

    def makeVertexList(r,**kwargs):
        '''
        Create a list of Vertex objects to be used in the MeshGraph. 
        Conveniently, the Vertex indices, stored for convenience within the 
        vertices themselves, correspond to their position within 
        vertexList. 
        Inputs:
          r: an (N,d) array of Vertex positions where N is number of vertices
        '''
        vertexList = []
        for i in range(np.shape(r)[0]):
          vertexList += Vertex(i,r[i])
        return vertexList

    def createVertexDictionary(self):
        vertexDict = dict()
        for i in self.vertexList:
            iNum = i.getNum()
            vertexDict[iNum] = i    
        return vertexDict    
        
    def showEdges(self):
      return self.edgeList 


    #Create a list of Edges in the triangulation
    def makeTriangulation(self):
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
      points = self.coordinateArray
      height = self.height
      #define cadre of height derivatives for generating curvatures/normals
      hx = grad(height,0)
      hy = grad(height,1)
      hxx = grad(grad(height,0),0)
      hyy = grad(grad(height,1),1)
      hxy = grad(grad(height,0),1)

      #define principal curvatures/normals
      def kminFunction(dx,dy,dxx,dxy,dyy,**kwargs):
        return (dxx - sqrt(dxy**2 + (dxx-dyy)**2)+dyy)/(2*sqrt(1+dx**2+dy**2))
      
      def kmaxFunction(dx,dy,dxx,dxy,dyy,**kwargs):
        return (dxx + sqrt(dxy**2 + (dxx-dyy)**2)+dyy)/(2*sqrt(1+dx**2+dy**2))
      
      def normalFunction(dx,dy,**kwargs):
        norm = sqrt(1+dx**2 + dy**2)
        return np.array([-dx/norm,-dy/norm,1/norm])

      def neighborhoodSize(kmax,kmin,**kwargs):
        #dynamically sized neighborhood dependent on principal curvatures.
        #this function should never be called when kmax or kmin = 0

        return 2*abs(kmax/kmin) #must be positive to compare to E. distances

      def bisectorSlope(point1,point2,normal):
        #finds slope & point of perpendicular bisector between point1, point2
        pointSlope = (point2 - point1)
        orthogonalSlope = np.cross(pointSlope,normal) #(3,) vector slope in-plane
        midpoint = (point1+point2)/2
        return orthogonalSlope, midpoint #testing for easier to read slopes
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

        if np.isclose(sx1*alpha + x01, sx2*beta + x02):
          #print('Intersection!')
          return np.array([alpha*sx1+x01, alpha*sy1+y01, alpha*sz1+z01])
        
        else:
          #print('no intersection :(')
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
        #checks if point B is a delaunay neighbor of r,
        #assuming point B is between (by angle) points A and C
        #coefficients of the perpendicular bisectors of r - p_alpha
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

        #define intersection point of perpendicular bisectors two and three,
        #other algorithm quantities 
        IAC = intersectionPoint(slopeA,slopeC,midpointA,midpointC)
        #check for parallel/no intersections
        if (len(IAC) == 1):
          if IAC[0] == None:
            #print('Parallel or no intersection.')
            return False

        slopeIB, intersectIB = slopeB, IAC #define LI1 as line parallel to bisector of line r1 passing through I23
        linepoint1 = slopeIB*(-1)+intersectIB
        linepoint2 = slopeIB*(1)+intersectIB

        rside = isLeft(linepoint1,linepoint2,r) #returns TRUE/FALSE
        mpBside = isLeft(linepoint1,linepoint2,midpointB) #returns TRUE/FALSE

        return rside == mpBside #if same side, they are D. neighbors 

      appendedSets = []
      for i in range(np.shape(points)[0]):
        #define current point and Euclidean distances between this point and
        #all other points
        r = points[i]
      
        distances = self.pointToAllDistance(r,points) #(N,) list of distances to our source point
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
        kmin = kminFunction(dx,dy,dxx,dxy,dyy)
        kmax = kmaxFunction(dx,dy,dxx,dxy,dyy)

        localNormal = normalFunction(dx,dy)
        kNeighborhood = neighborhoodSize(kmax,kmin)

        #average number of nearest neighbors should be ~6 for a decently large
        #triangulation w/poisson sampled points (implement!). So if we check a few
        #points than that, we should be checking approx. all relevant points.
        #Should really only run the risk of being a little slow for small data
        #sets, or miss a neighbor or two in larger ones...
        #that, and we'll find a few spurious edges from considering too many points.
        if kmax*kmin == 0:
          avgNeighborNumber = 7
          neighborIndices = jnp.copy(np.argpartition(numpyDistances,avgNeighborNumber))[:avgNeighborNumber]

        else: 
          neighborIndices = jnp.where(distances < kNeighborhood)[0]

        neighborPoints = points[neighborIndices]
        neighborVectors = neighborPoints - r 

        normalOverlap = jnp.dot(neighborVectors,localNormal) #(N,) scalar dot products

        #neighbor vectors come out as (1,N,d) shaped arrays, which we acknowledge in indices later
        projectedNeighborVectors = neighborVectors - jnp.einsum('i,j->ij',np.dot(neighborVectors,localNormal),localNormal)
        projectedNeighborPoints = r + projectedNeighborVectors 
        projectedNeighborAngles = jnp.arctan2(projectedNeighborVectors[:,1],projectedNeighborVectors[:,0])

        #sort points by lowest to highest angle (angles range [-pi,pi])
        indexOrder = projectedNeighborAngles.argsort()

        sortedNeighbors = neighborPoints[indexOrder]
        sortedProjectedNeighbors = projectedNeighborPoints[indexOrder]
        sortedNeighborVectors = projectedNeighborVectors[indexOrder]
        sortedNeighborDistances = np.linalg.norm(sortedNeighborVectors,axis=1)
        sortedIndices = neighborIndices[indexOrder] #necessary to draw edges 
        
        #now find Delaunay neighbors. 
        verticesToTest = list(range(jnp.shape(sortedProjectedNeighbors)[0]))

        #loop initializers.
        bIndex = np.argmin(sortedNeighborDistances) #closest point, guaranteed to be a neighbor
        startingB = verticesToTest[bIndex]
        countedSets = []

        while len(verticesToTest) > 2: 

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
            #check if we're ready to break because we've checked all the candidates
            if set(verticesToTest).issubset(set(countedSets)): 
              #true if counted sets has all elements of the vertices to test -- 
              #it can also have a few of the removed vertices!
              print('Checked all possibilities for candidate ', i)

              break
            countedSets.append(verticesToTest[bIndex])
            edgeLength = euclideanDistance(r,sortedNeighbors[B])
            currentVertex = sortedIndices[verticesToTest[bIndex]].item() #.item() prevents an array from being returned
            pointSet = {i,currentVertex}
            if pointSet != {i}: #this should basically never stop an append unless undersampling
              if pointSet not in appendedSets:
                edgeList.append(Edge(pointSet,edgeLength))
                appendedSets.append(pointSet)

            bIndex += 1

          elif (not isNeighbor): #this else CANNOT happen for the first vertex as a rule
            verticesToTest.remove(verticesToTest[bIndex])
            bIndex -= 1
            
        if len(verticesToTest) <= 2:
              print('Insufficient vertices for further evaluation at candidate ', i)
      return edgeList



class Face:
    '''
    Class that stores faces (the triangles) of a triangular mesh. 
    -----
      verts = vertices of the triangle, [3,] list of Vertex objects
      edges = edges of the triangle, [3,] list of Edge objects (can be 
        generated from vertices)
      area = triangle area, generated using Heron's formula. I could probably
        get rid of this if efficiency is really, really important... no idea
        when it's used
    '''
    def __init__(self,verts,edges = [],area = None):
      if edges == []:
        #there has GOT to be a way to do this in fewer lines
        pair1 = [verts[0],verts[1]]
        pair2 = [verts[0],verts[2]]
        pair3 = [verts[1],verts[2]]
        dist1 = np.linalg.norm(verts[0].point()-verts[1].point())
        dist2 = np.linalg.norm(verts[0].point()-verts[2].point())
        dist3 = np.linalg.norm(verts[1].point()-verts[2].point())
        edge1 = Edge(pair1,dist1)
        edge2 = Edge(pair2,dist2)
        edge3 = Edge(pair3,dist3)
        edges = [edge1,edge2,edge3]

      if area == None:
        #calculate area using heron's formula
        a = edges[0].getLength()
        b = edges[1].getLength()
        c = edges[2].getLength()
        p = (a+b+c)/2
        area = np.sqrt(p*(p-a)*(p-b)*(p-c))

      self.verts = verts
      self.edges = edges
      self.area = area

    def getVerts(self):
      return self.verts
    def getEdges(self):
      return self.edges
    def getArea(self):
      return self.area

