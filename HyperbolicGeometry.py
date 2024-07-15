"""
Affine plane: A²
Affine point: A[x, y]
Affine line: ax + by = c <-> (a:b:c)

To deccribe projective plane P² we need
3-dim vectorspace V³ (x, y, z)

Projective points: [x:y:z] is a line that passes through the origin
point on the viewing plane: if z != 0: [x:y:z] = [x/z:y/z:1]
viewing plane coordinates: x = x/z, y=y/z
horizontal points in the xy plane: if z = 0: [x:y:0] = [x,y]

Projective lines: (l:m:n) is a plane that passes through the origin
hyperbolic-line: lx+my-nz = 0 <-> L = (l:m:n)
eliptic-line: lx+my+nz = 0 <-> L = (l:m:n)
if (l,m) != (0,0) then when z = 1
the plane lx+my-nz=0 simplifies to: lx+my=n
because the line intersects the viewing plane
"""
from typing import TypedDict
from typing import NamedTuple
from fractions import Fraction
from mylib.num import factor, sieve, divisors, is_prime
class HybPoint(NamedTuple):
  x: Fraction
  y: Fraction
  z: Fraction

class HybLine(NamedTuple):
  l: Fraction
  m: Fraction
  n: Fraction

class HybSide(NamedTuple):
  a1: HybPoint
  a2: HybPoint

class HybVertex(NamedTuple):
  L1: HybLine
  L2: HybLine

class HybCouple(NamedTuple):
  a: HybPoint
  L: HybLine
class HybTriangle(NamedTuple):
  a1: HybPoint
  a2: HybPoint
  a3: HybPoint
class HybTrilateral(NamedTuple):
  L1: HybLine
  L2: HybLine
  L3: HybLine







class PlanarPoint(NamedTuple):
  X: float
  Y: float

class PlanarLine(NamedTuple):
  l: float
  m: float
  n: float

def join(a1: HybPoint2D, a2: HybPoint2D):
  l = (a1.y * a2.z) - (a2.y * a1.z)
  m = (a1.z * a2.x) - (a2.z * a1.x)
  n = (a2.x * a1.y) - (a1.x * a2.y)
  return HybLine2D()

def meet(L1, L2):
  return 1

def is_null(prop):
  return 1

def lies_on_line(a, L):
  return 1

def cross_prod(a1, a2):
  return 1

def is_collinear(list_of_points):
  return 1


def planar_point(a):
  return 1

def planar_line(L):
  return 1


def is_perp(a1, a2):
  return 1


def is_concurrent(L1, L2):

def pass_through_point(self, a:HyperbolicPoint3D):
  return (self.l * a.x) + (self.m * a.y) - (self.n * a.z) == 0






def has_equal_proportion(p1, p2):
  if len(p1) == len(p2):
    dim = len(p1)
    for i in range(dim):
      i1 = i
      i2 = (i + 1) % dim
      check = (p1[i1] * p2[i2]) * (p2[i1] - p2[i2]) == 0
      if check is False:
        return False
    return True




class HyperbolicPoint3D:

  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z

  def __eq__(self, a):
    """
    :type a: HyperbolicPoint3D
    """
    return [self.x, self.y, self.z] == [a.x, a.y, a.z]

  def dual(self):
    return HyperbolicLine3D(self.x, self.y, self.z)

  def lies_on_line(self, L):
    """
    :type L: HyperbolicLine3D
    """
    return (L.l * self.x) + (L.m * self.y) - (L.n * self.z) == 0

  def is_null(self):
    return (pow(self.x, 2) + pow(self.y, 2)) - pow(self.z, 2) == 0

  def is_perp(self, a):
    """
     :type a: HyperbolicPoint3D
     :returns: bool
     """
    x1, y1, z1, x2, y2, z2 = self.x, self.y, self.z, a.x, a.y, a.z
    return ((x1*x2) + (y1*y2) - (z1*z2)) == 0

  def planar_point(self):
    return PlanarPoint(self.x/self.z, self.y/self.z)

  def join(self, a):
    """
    :type a: HyperbolicPoint3D
    :returns: HyperbolicLine3D
    """
    if self != a:
      x1, y1, z1, x2, y2, z2 = self.x, self.y, self.z, a.x, a.y, a.z
      nx = ((y1 * z2) - (y2 * z1))
      ny = ((z1 * x2) - (z2 * x1))
      nz = ((x2 * y1) - (x1 * y2))
      return HyperbolicLine3D(nx, ny, nz)
    else:
      raise ArithmeticError("not distinct")

  @staticmethod
  def cross_prod(a1, a2):
    """
    :type a1: HyperbolicPoint3D
    :type a2: HyperbolicPoint3D
    """
    x1, y1, z1 = a1.x, a1.y, a1.z
    x2, y2, z2 = a2.x, a2.y, a2.z
    x3 = (y1 * z2) - (y2 * z1)
    y3 = (z1 * x2) - (z2 * x1)
    z3 = (x2 * y1) - (x1 * y2)
    return HyperbolicPoint3D(x3, y3, z3)

  @staticmethod
  def collinear(lp):



  @staticmethod
  def is_collinar(a1, a2, a3):
    """
      :type a1: HyperbolicPoint3D
      :type a2: HyperbolicPoint3D
      :type a3: HyperbolicPoint3D
    """
    l = join(a1, a2)
    x1, y1, z1 = a1.x, a1.y, a1.z
    x2, y2, z2 = a2.x, a2.y, a2.z
    x3, y3, z3 = a3.x, a3.y, a3.z
    return (x1*y2*z3) - (x1*y3*z2) + (x2*y3*z1) - (x3*y2*z1) + (x3*y1*z2) - (x2*y1*z3) == 0
class HyperbolicLine3D:

  def __init__(self, l, m, n):
    self.l = l
    self.m = m
    self.n = n

  def __eq__(self, L):
    """
    :type L: HyperbolicLine3D
    """
    return [self.l, self.m, self.n] == [L.l, L.m, L.n]

  def dual(self):
    return HyperbolicPoint3D(self.l, self.m, self.n)

  def pass_through_point(self, a:HyperbolicPoint3D):
    return (self.l * a.x) + (self.m * a.y) - (self.n * a.z) == 0

  def is_null(self):
    return (pow(self.l, 2) + pow(self.m, 2)) - pow(self.n, 2) == 0

  def is_perp(self, a):
    """
     :type a: HyperbolicLine3D
     """
    l1, m1, n1, l2, n2, m2 = self.l, self.m, self.n, a.l, a.m, a.n
    return ((l1*l2) + (m1*m2) - (n1*n2)) == 0

  def meet(self, L):
    """
    :type L: HyperbolicLine3D
    :returns a: HyperbolicPoint3D
    """
    if self == L:
      raise ArithmeticError("not distinct")
    l1,l2,m1,m2,n1,n2 = self.l, L.l, self.m, L.m, self.n, L.n
    l3 = (m2*n2) - (m2*n1)
    m3 = (n1*l2) - (n2*l1)
    n3 = (l2*m1) - (l1*m2)
    return HyperbolicPoint3D(l3, m3, n3)

  @staticmethod
  def is_concurrent(a1, a2, a3):
    """
      :type a1: HyperbolicPoint3D
      :type a2: HyperbolicPoint3D
      :type a3: HyperbolicPoint3D
    """
    x1, y1, z1 = a1.x, a1.y, a1.z
    x2, y2, z2 = a2.x, a2.y, a2.z
    x3, y3, z3 = a3.x, a3.y, a3.z
    return (x1 * y2 * z3) - (x1 * y3 * z2) + (x2 * y3 * z1) - (x3 * y2 * z1) + (x3 * y1 * z2) - (x2 * y1 * z3) == 0




def has_duality(point, line):
  return has_equal_proportion(point, line)

def point_lies_on_line(point, line):
  l, m, n, x, y, z = point[0], point[1], point[2], line[0], line[1], line[2]
  return (l*x + m*y) - n*z == 0

def line_pass_through_point(point, line):
  return point_lies_on_line(point, line)

def is_null(prop3):
  return pow(prop3[0], 2) + pow(prop3[1], 2) - pow(prop3[2], 2) == 0

def is_perp(hyperbolic_point1, hyperbolic_point2):
  p1 = hyperbolic_point1
  p2 = hyperbolic_point2
  a1, b1, c1, a2, b2, c2 = p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]
  return (a1 * a2) + (a1 * a2) - (a1 * a2) == 0

def planar_point(hyperbolic_point):
  a = hyperbolic_point
  plx = a[0] / a[2]
  ply = a[1] / a[2]
  return [plx, ply]

def planar_line(hypberbolic_line):
  L = hypberbolic_line
  X, Y = L[0] / L[2]

