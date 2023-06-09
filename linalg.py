#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from fractions import Fraction as frac
from numbers import Rational as Q
import math


class Mtx:
    
 @staticmethod
 def getcol(i,mtx):
   return [row[i] for row in mtx]  
       
 @staticmethod
 def getid(n,m): 
    ret = [[0] * m] * m
    for i in range(m):
      for j in range(n):
        v = 1 if i == j else 0
        ret[i][j] = v
    return ret  
       










 

class SqrMtx:
  @property
  def I2():
    return SqrMtx([[1,0],[1,0]])
  @property
  def G2():
    return SqrMtx([[1,0],[0,-1]])
  @property
  def R2():
    return SqrMtx([[0,1],[1,0]])
  @property
  def B2():
    return SqrMtx([[0,-1],[1,0]])    
  @property
  def O2():
    return SqrMtx([[0,0],[0,0]])
  @property
  def Shear2():
    return SqrMtx([[1,1],[0,1]])    
    
  @staticmethod
  def zeroLit(dim):
    return [[0] * dim] * dim

  @staticmethod
  def getcol(i,mtx):
    return [row[i] for row in mtx]  
        
  @staticmethod
  def idmtx(n):
    ret = [[0] * n] * n
    for i in range(n):
      ret[i][i] = 1    
    return ret        
    
  @staticmethod
  def validLiteral(literal):
   if type(literal) is list:
    #print('list check passed')
    shape = np.shape(literal)
    if len(shape) == 2 and shape[0] == shape[1]:
      #print('shape check passed')  
      for i,row in enumerate(literal):
        if any([isinstance(v,Q) == False for v in row]): return False
      return True
    else: return False
   else: return False
  
  #rotation matrix in the unit circle 
  @staticmethod
  def Ptheta(theta):
    s = math.sin(theta)
    c = math.cos(theta)
    return SqrMtx([[c,-s],[s,c]]) 
   
       
  def __init__(self,inp):
    if SqrMtx.validLiteral(inp):
      self.dim = np.shape(inp)[0]
      self.literal = inp
      self.cols = [SqrMtx.getcol(i, inp) for i in range(self.dim)]
              
      

    else: raise('type error!')
    
  def det(self):
    return int(round(np.linalg.det(np.array(self.literal))))

  def minor(self,indtup):
    mtx = self.literal
    ret = [[v for j,v in enumerate(mtx[i]) if j != indtup[1]] for i,r in enumerate(mtx) if i != indtup[0]]  
    return ret[0][0] if self.dim == 2 else ret

  
  def inverse2d(self):
    #a = self.minor((0,0))
    #b = self.minor((1,0))
    #c = self.minor((0,1))
    #d = self.minor((1,1))
    a = self.literal[0][0]
    b = self.literal[1][0]
    c = self.literal[0][1]
    d = self.literal[1][1]
    return SqrMtx([[d,-1 * b],[a,-1 * c]])


  def __mul__(self,val):
    lit = SqrMtx.zeroLit(self.dim) 
    if isinstance(val,Q): 
      for ri,row in enumerate(self.literal):
        for ei,ent in enumerate(row):
         lit[ri][ei] = ent * val 
      return SqrMtx(lit)  
  
    elif isinstance(val,SqrMtx) and val.dim == self.dim:
      for ri,row in enumerate(self.literal):
        for ei,ent in enumerate(row):
          lit[ri][ei] = ent * val[ri][ei]
      return SqrMtx(lit)
        
    elif isinstance(val,Vector) and val.dim == self.dim:
      for ri,row in enumerate(self.literal):
        lit[ri] = sum([tup[0] * tup[1] for tup in zip(row,val.arr)])
      return Vector(lit)
    else: raise('type error!')
          
  def __add__(self,sqrmtx):
    if isinstance(sqrmtx,SqrMtx) and sqrmtx.dim == self.dim:
      lit = SqrMtx.zeroLit(self.dim)  
      for ri,row in enumerate(self.literal):
        for ei,ent in enumerate(row):
          lit[ri][ei] = ent + sqrmtx[ri][ei]
      return SqrMtx(lit)
    else: raise('type error!')

  def __sub__(self,sqrmtx):
    if isinstance(sqrmtx,SqrMtx) and sqrmtx.dim == self.dim:
      lit = SqrMtx.zeroLit(self.dim)  
      for ri,row in enumerate(self.literal):
        for ei,ent in enumerate(row):
          lit[ri][ei] = ent - sqrmtx[ri][ei]
      return SqrMtx(lit)
    else: raise('type error!')
    
    
    
    
class Vector:
  @staticmethod
  def isCompatible(vec1,vec2):
    return isinstance(vec1,Vector) and isinstance(vec2,Vector) and vec1.dim == vec2.dim   
  
  def sameTail(vec1,vec2,checkInput=True):
    if checkInput:
      if Vector.isCompatible(vec1,vec2):  
        for i in range(vec1.dim - 1):
          if vec1[i] != vec2[i]: return False    
        return True
      return False
    for i in range(vec1.dim - 1):
      if vec1[i] != vec2[i]: return False    
    return True
  # rational rotation on the unit circle  
  @staticmethod
  def rotationVector2(h):
    ch = (1 - pow(h,2)) / (1 + pow(h,2))
    sh = (2 * h) / (1 + pow(h,2))
    return Vector([ch,sh])

  @staticmethod
  def area(vec1,vec2,checkInput=True):
   if checkInput:   
     if Vector.isCompatible(vec1, vec2) and vec1.dim == 2:
       a = vec1.arr[0]
       b = vec1.arr[1]
       c = vec2.arr[0]
       d = vec2.arr[1]
       return (a * d) - (b * c)
     else: raise('not computeable')
   a = vec1.arr[0]
   b = vec1.arr[1]
   c = vec2.arr[0]
   d = vec2.arr[1]
   return (a * d) - (b * c)   
  
  @staticmethod
  def joiningVector(vec1,vec2,checkInput=True):
   if checkInput:   
     if Vector.isCompatible(vec1, vec2) and Vector.sameTail(vec1, vec2):
       return vec1 - vec2
     else: raise('not computeable')
   return vec1 - vec2 
 
  @staticmethod
  def affineComb(vec1,vec2,l,checkInput=True):
    if checkInput:
      if Vector.isCompatible(vec1, vec2) and Vector.sameTail(vec1, vec2,False) and isinstance(l,Q):
        jvec = vec1 - vec2
        return vec1 + (l * jvec)
      else: raise('not computeable!')
    jvec = vec1 - vec2
    lvec = l * jvec
    return vec1 + lvec

  @staticmethod
  def paralellorgram(v,u):
   if Vector.sameTail(v, u):   
     v1 = v
     v2 = 2 * v
     u1 = u
     u2 = 2 * u
     d1 = u + v
     d2 = u - v
     return {'v1':v1,'v2':v2,'u1':u1,'u2':u2,'d1':d1,'d2':d2}
   else: raise('not computeable')
   
  @staticmethod
  def y1y2(m,x1x2):
    if isinstance(m,SqrMtx) and m.dim == 2 and isinstance(x1x2,Vector) and x1x2.dim == 2:
      return m * x1x2     
    else: raise('type error!')
    
  def z1z2(m1,m2,x1x2):
    if all([isinstance(v,SqrMtx) and v.dim == 2 for v in [m1,m2]]) and isinstance(x1x2,Vector) and x1x2.dim == 2:    
    #if isinstance(m1,SqrMtx) and m1.dim == 2 and and isinstance(x1x2,Vector) and x1x2.dim == 2:
      return (m1 * m2) * x1x2      
    else: raise('type error!')  
  
  def __init__(self,arr):
    if type(arr) is list and len(np.shape(arr)) == 1 and all([isinstance(v,Q) for v in arr]):
      self.arr = arr
      self.dim = len(arr)
    else: raise('type error!')
       
    
  def isNonZero(self):
   return any([v != 0 for v in self.arr])
    
  def dot(self,vect):
    if isinstance(vect,Vector) and vect.dim == self.dim:
      return sum([tup[0] * tup[1] for tup in zip(self.arr,vect.arr)])
    else: raise('type error!')
  
  def norm(self):
    return math.sqrt(sum([pow(v,2) for v in self.arr]))

  
  def isPerp(self,vect):
    if isinstance(vect,Vector) and vect.dim == self.dim:
      if self.isNonZero() and vect.isNonZero(): 
        return self.dot(vect) == 0;
      else: return False;
    else: raise('type error!')
  
  def isColinear(self,vect):
    if isinstance(vect,Vector) and vect.dim == self.dim:
      if self.isNonZero() == False or vect.isNonZero() == False: return False  
      else: return abs(self.dot(vect)) == self.norm * vect.norm
    else: raise('type error!')
     
    
  def __mul__(self,inp):
    if isinstance(inp,Vector) and inp.dim == self.dim:
     na = list([tup[0] * tup[1] for tup in zip(self.arr,inp.arr)])   
     print(na)
     return Vector(na)    
    
    elif isinstance(inp,Q):
     return Vector(list([inp * v for v in self.arr]))    
    
    else: raise('type error!')
    
  def __neg__(self):
    arr = list([-v for v in self.arr])
    return Vector(arr)
    
  def __add__(self,vect):
    if isinstance(vect,Vector) and self.dim == vect.dim:
      arr = list([tup[0] + tup[1] for tup in zip(self.arr,vect.arr)])
      return Vector(arr)
    else: raise('type error!')
    
  def __sub__(self,vect):  
    if isinstance(vect,Vector) and self.dim == vect.dim:
      arr = list([tup[0] - tup[1] for tup in zip(self.arr,vect.arr)])
      return Vector(arr)
    else: raise('type error!')      
    
class VectorSpace:
  #check if input is a one dimenional list where all elements are rationals.   
  @staticmethod
  def validLiteral(arr):
    if type(arr) is list and len(np.shape(arr)) == 1 and all([isinstance(v,Q) for v in arr]): 
        return True
    else: return False
    
  def __init__(self,arr):
    if VectorSpace.validLiteral(arr):   
      self.dim = len(arr)
      self.basis_vectors = []
      # turn input array to a list of basis vectors for every dimensional axis
      for i,val in enumerate(arr):
        # generate a list of zeroes of length equal to the number of dimensions of the vector space
        v = [0] * self.dim
        #set the value of arr[i] at the corresponding index of a basis vector for the same dimensional axis
        v[i] = val
        self.basis_vectors.append(Vector(v))
    else: raise('type error!')

  #multiply this matrix with any vector V1 existing in another VectorSpace w to get 
  #another vector V2 which is V1 existing in this VectorSpace   
  def get_transformation_matrix(self, w):
    #check if input is a VectorSpace with the same dimensions
    if isinstance(w,VectorSpace) and w.dim == self.dim:
    #generate a square matrix with the same number of rows and elements equal to the number of dimensions in the VectorSpace  
      A = [[0] * self.dim] * self.dim 
      for i in range(self.dim):
        for j in range(self.dim):
          #populate the matrix with with the dot product between every basis vector of the two VectorSpaces
          A[i][j] = np.dot(w.basis_vectors[i].arr, self.basis_vectors[j].arr)
      return SqrMtx(A)  
    else: raise('type erro!')    
    
class Point:  
  @staticmethod 
  def isCompatible(point_array):
    return all([isinstance(v,Point) for v in point_array]) and len(set([v._dim for v in point_array])) == 1    
  
  @staticmethod
  def isColinear(point_array):
    if Point.isCompatible(point_array) and len(point_array) >= 3:  
     pa = np.array([v._arr for v in point_array])
     rank = np.linalg.matrix_rank(pa)
     return rank <= 1    
    else: return False  
  
  @staticmethod
  def validLiteral(lit):
    return type(lit) is list and np.shape(lit)[0] == 1 and all([isinstance(v,Q) for v in lit])
        
      
  def __init__(self,arr):
    if Point.validLiteral(arr):
      self._arr = arr
      self._dim = len(arr)
      self._space = VectorSpace([1] * self._dim)
    else: raise('type error!')
          
  #def isColinear(self,pointArray):
    #if isinstance(p2,Point) and p2._dim == self._dim:      
        
class Lin:
    
  def __init__(self,p1,p2):
    if Point.isCompatible([p1,p2]):
      self.p1 = p1 if isinstance(p1,Point) else Point(p1)
      self.p2 = p2 if isinstance(p2,Point) else Point(p2)
      self.dim = p1.dim
    else: raise('type error!')
  
  def is_distinct(self,line):
    if isinstance(line,Lin):
      return len(set(all([v._arr for v in [self.p1,self.p2,line.p1,line.p2]]))) > 2   
    else: raise('error')  
  def intersects(self,line):
    return True
             
  def is_scew(self,line):
    return True
    
  def is_paralell(self,line):
    return True      


class Line2d:
  @staticmethod
  def solve_line_eq_general(a,b,c):
    if a == 0 and b == 0:
        raise ValueError("Invalid equation: a and b cannot both be zero.")
    elif a == 0:
        y = c / b
        return (0, y)
    elif b == 0:
        x = c / a
        return (x, 0)
    else:
      x = c / a
      y = (c - a * x) / b
      return [x, y]
  
  @staticmethod
  def solve_line_eq_param(A,B,l):
    P = [-l*A[i] + B[i] for i in range(len(A))]
    return P

  @staticmethod
  def solve_line_eq(m, b, x):
    y = m * x + b
    return y  
            
  @staticmethod
  def findIntersect(l1,l2):
    #m = SqrMtx(ml)
    m = SqrMtx([ [ l1[0], l1[1] ],[ l2[0], l2[1] ] ])
    d = (m.literal[0][0] * m.literal[1][1]) - (m.literal[1][0] * m.literal[0][1])
    mi = m.inverse2d()
    print(mi.literal)
    v = Vector([l1[2],l2[2]])
    res = (mi * frac(1,d)) * v
    #print(v.arr)
    #print(res.arr)
    return res
    #xy = (1/d * minors) * Vector([l1[2],l2[2]])
    #print(xy)











    
