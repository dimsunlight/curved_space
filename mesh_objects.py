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
from mesh_operations import removeExcessEdges, makeTriangulation

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
        self.edges = edges #effectively list of vertices we're nearest neigbors to
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
        #returns a list of edge objects -- test?
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

        self.height = heightFunction
        self.coordinateArray = coordinateArray #array
        self.vertexList = self.makeVertexList(self.coordinateArray)  
        self.edgeList,self.vertexList = makeTriangulation(
                                               self.coordinateArray,
                                               self.height,
                                               self.vertexList)
        print('Beginning pruning procedure...')
        self.edgeList,self.vertexList = removeExcessEdges(
                                              self.edgeList,
                                              self.coordinateArray,
                                              self.vertexList)
        
        self.vertexDictionary = self.createVertexDictionary()
        self.faceList = faces
        

    def defaultHeight(self,x,y,**kwargs):
      '''
      written in terms of x and y so we can take automatic derivatives w.r.t. 
      both x and y.   
      '''
      return 0 

    def makeVertexList(self,r,**kwargs):
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
          vertexList.append(Vertex(i,r[i]))
        return vertexList

    def createVertexDictionary(self):
        vertexDict = dict()
        for i in self.vertexList:
            iIndex = i.getIndex()
            vertexDict[iIndex] = i    
        return vertexDict    
        
    def showEdges(self):
      return self.edgeList 

    def pruneEdges(self):
      return removeExcessEdges(self.edgeList,self.coordinateArray,
                               self.vertexList)

