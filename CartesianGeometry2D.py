import math
import numpy as np
from fractions import Fraction
import mylib.comb as comb

def pyth_tripple(m,n):
  m2, n2 = m**2, n**2
  a = (m2 - n2) ** 2
  b = (2*m*n) ** 2
  c = (m2 + n2) ** 2
  return [[a, b, c], a+b, c]


print(pyth_tripple(2,1))
def array_check_all(a, predicate):
  for el in a:
    if predicate(el) is False:
      return False
  return True

def array_check_any(a, predicate):
  for el in a:
    if predicate(el) is True:
      return True
  return False

def matrix_check_all(m, predicate):
  for row in m:
    if array_check_all(row, predicate) is False:
      return False
  return True

def matrix_check_any(m, predicate):
  for row in m:
    if array_check_all(row, predicate) is True:
      return True
  return False


def is_valid_number(x):
  return True if isinstance(x, int) or isinstance(x, Fraction) else False

def div(nom, dnom):
  if nom // dnom - nom / dnom == 0:
    return nom // dnom
  else:
    return Fraction(nom, dnom)

class Point2D:
  @staticmethod
  def check_arg_point(x):
    if isinstance(x, Point2D) is False:
      raise ArithmeticError("expected argument to be instance of class: Point2D")
    else:
      return True

  def __init__(self, x, y):
    if is_valid_number(x) is False or is_valid_number(y) is False:
      raise ArithmeticError("x and y must be integer or rational")
    self.x = x if is_valid_number(x) else 0
    self.y = y if is_valid_number(y) else 0


  def point_on_two_lines(self, l1, l2):
    '''
    If the lines l1 and l2 are not parallel, then there
    is a unique point A ≡ l1 l2 which lies on them both.
    If l1 ≡ ha1 : b1 : c1 i and l2 ≡ a2 : b2 : c2  then:
    x = b1 c2 − b2 c1 / a1 b2 − a2 b1
    y = c1 a2 − c2 a1 / a1 b2 − a2 b1
    '''
    a1,b1,c1,a2,b2,c2 = l1.a, l1.b, l1.c, l2.a, l2.b, l2.c
    dnom = (a1 * b2) - (a2 * b1)
    x = div((b1 * c2) - (b2 * c1), dnom)
    y = div((c1 * a2) - (c2 * a1), dnom)
    return Point2D(x, y)

  def lies_on_line(self, l):
    '''
    The point A ≡ [x, y] lies on the line l ≡ <a : b : c>,
    or equivalently the line l passes through the point A,
    precisely when ax + by + c = 0.
    '''
    return (l.a * self.x) + (l.b * self.y) + l.c == 0


  @staticmethod
  def is_collinear(list_of_points):
    '''
    The points [x1 , y1 ], [x2 , y2 ] and [x3 , y3 ] are collinear precisely when
      x1 y2 − x1 y3 + x2 y3 − x3 y2 + x3 y1 − x2 y1 = 0.
    :param list_of_points:
    :return:
    '''
    c = len(list_of_points)
    if c == 3:
      l = Line2D.line_through_points(list_of_points[0], list_of_points[1])
      return array_check_all(list_of_points, lambda p: p.lies_on_line(l))
    else:
      raise ArithmeticError("argument must have at least three elements")




  @staticmethod
  def Foot_of_altitude(A, l):
    '''
    For any point A ≡ [x, y] and any non-null line l ≡ <a : b : c>,
    the altitude n from A to l intersects l at the point:
      F ≡ [x,y]
      x =  b²x - aby - ac  / a² + b²
      y = -abx + a²y - bc /  a² + b²
    '''
    Point2D.check_arg_point(A)
    Line2D.check_arg_line(l)
    if l.is_null() is not True:
      a,b,c,x,y = l.a, l.b, l.c, A.x, A.y
      sqr_a,sqr_b = pow(a, 2), pow(b, 2)
      dnom = sqr_a + sqr_b
      nx = div((sqr_b * x) - (a * b * y) - (a * c), dnom)
      ny = div(-(a * b * x) + (sqr_a * y) - (b * c), dnom)
      return Point2D(nx, ny)
    else:
      raise ArithmeticError("l can not be null")

  @staticmethod
  def is_distinct(p1, p2):
    Point2D.check_arg_point(p1)
    Point2D.check_arg_point(p2)
    if p1.x == p2.x and p1.y == p2.y:
      return False
    else:
      return True

  @staticmethod
  def Q(p1, p2):
    Point2D.check_arg_point(p1)
    Point2D.check_arg_point(p2)
    return (abs(p1.x - p2.x) ** 2) + (abs(p1.y - p2.y) ** 2)

  @staticmethod
  def Midpoint(p1, p2):
    Point2D.check_arg_point(p1)
    Point2D.check_arg_point(p2)
    nx = div(p1.x + p2.x, 2)
    ny = div(p1.y + p2.y, 2)
    return Point2D(nx, ny)

  @staticmethod
  def forms_null_line(p1, p2):
    Point2D.check_arg_point(p1)
    Point2D.check_arg_point(p2)
    if Point2D.is_distinct(p1, p2):
      return Point2D.Q(p1, p2) == 0

  def __str__(self):
    return "x:{a} y:{b}".format(a=self.x,b=self.y)

  def __repr__(self):
    return "x:{a} y:{b}".format(a=self.x, b=self.y)



class Line2D:
  @staticmethod
  def check_arg_line(x):
    if isinstance(x, Line2D) is False:
      raise ArithmeticError("expected argument to be instance of class: Point2D")
    else:
      return True

  #A1 A2 = hy1 − y2 : x2 − x1 : x1 y2 − x2 y1 i .
  def __init__(self,a, b, c):
    if is_valid_number(a) and is_valid_number(b) and is_valid_number(c):
      self.a = a
      self.b = b
      self.c = c
    else:
      raise ArithmeticError("arguments to Line2D constructor must be integer or rational")

  def pass_through_point(self, p):
    return (self.a * p.x) + (self.b * p.y) + self.c == 0

  def prop(self):
    return abs(self.a - self.b)

  def is_null(self):
    return (self.a ** 2) + (self.b ** 2) == 0

  def is_distinct(self, l):
    if self.a == l.a and self.b == l.b and self.c == l.c:
      return False
    else:
      return True

  def is_central(self):
    return self.c == 0

  @staticmethod
  def is_concurrent(list_of_lines):
    c = len(list_of_lines)
    if c >= 3:
      p = Line2D.intersection_point(list_of_lines[0], list_of_lines[1])
      return array_check_all(list_of_lines, lambda l: l.pass_through_point(p))
    else:
      raise ArithmeticError("argument must have at least three elements")

  def is_parallel_to(self, l):
    Line2D.check_arg_line(l)
    return self.prop() == l.prop()

  @staticmethod
  def is_parallel(list_of_lines):
    if isinstance(list_of_lines, list) and array_check_all(list_of_lines, lambda l: isinstance(l,Line2D)):
      p = list_of_lines[0].prop()
      return array_check_all(list_of_lines, lambda l: l.prop() == p)
    else:
      errmsg = "call to static method: is_parallel expected argument of type [Line2D], got:" + str(type(list_of_lines))
      raise ArithmeticError(errmsg)

  @staticmethod
  def intersection_point(l1, l2):
    '''
      if two lines are not parallel, there exist a unique point which lies on them both where:
        x = ((b1 * c2) − (b2 * c1)) / ((a1 * b2) − (a2 * b1))
        y = ((c1 * a2) - (c2 * a1)) / ((a1 * b2) - (a2 * b1))
    '''
    Line2D.check_arg_line(l1)
    Line2D.check_arg_line(l2)
    if l1.prop() != l2.prop():
      xnom = (l1.b * l2.c) - (l2.b * l1.c)
      xdnom = (l1.a * l2.b) - (l2.a * l1.b)
      ynom = (l1.c * l2.a) - (l2.c * l1.a)
      ydnom = (l1.a * l2.b) - (l2.a * l1.b)
      x = div(xnom,xdnom)
      y = div(ynom,ydnom)
      return Point2D(x, y)

  def is_perpendicular_to(self, l):
    '''
    The lines
    l1 ≡ a1 : b1 : c1 and l2 ≡ a2 : b2 : c2 are perpendicular precisely when:
    a1a2 + b1b2 = 0
    or equivalently when a1 : b1 = −b2 : a2.
    '''
    Line2D.check_arg_line(l)
    return (self.a * l.a) + (self.b * l.b) == 0

  @staticmethod
  def is_perpendicular(l1, l2):
    '''
    The lines
    l1 ≡ a1 : b1 : c1 and l2 ≡ a2 : b2 : c2 are perpendicular precisely when:
    a1a2 + b1b2 = 0
    or equivalently when a1 : b1 = −b2 : a2.
    '''
    Line2D.check_arg_line(l1)
    Line2D.check_arg_line(l2)
    return (l1.a * l2.a) + (l1.b * l2.b) == 0


  @staticmethod
  def line_through_points(p1, p2):
    Point2D.check_arg_point(p1)
    Point2D.check_arg_point(p2)
    if Point2D.is_distinct(p1, p2):
      na = abs(p1.y - p2.y)
      nb = abs(p2.x - p1.x)
      nc = abs((p1.x * p2.y) - (p2.x * p1.y))
      return Line2D(na, nb, nc)
    else:
      raise ArithmeticError("arguments must be distinct")

  @staticmethod
  def K(A, l):
    '''
    For any point A ≡ [x, y] and any line l ≡ <a : b : c>
    there is a unique line k, called the parallel through A to l,
    which passes through A and is parallel to l, namely:

        k = <a : b : −ax − by>
    '''
    Point2D.check_arg_point(A)
    Line2D.check_arg_line(l)
    return Line2D(l.a,l.b,(-l.a * A.x) - (l.b * A.y))

  @staticmethod
  def Alt_to_line(A, l):
    '''
    Theorem 7 (Altitude to a line)
    For any point A ≡ [x, y] and any line l ≡ <ha : b : c>
    there is a unique line n, called the altitude from A to l,
    which passes through A and is perpendicular to l, namely:

        n = <−b : a : bx − ay>
    '''
    Point2D.check_arg_point(A)
    Line2D.check_arg_line(l)
    return Line2D(-l.b, l.a, (l.b * A.x) - (l.a * A.x))

  @staticmethod
  def Foot_of_altitude(A, l):
    '''
    For any point A ≡ [x, y] and any non-null line l ≡ <a : b : c>,
    the altitude n from A to l intersects l at the point:
      F ≡ [x,y]
      x =  b²x - aby - ac  / a² + b²
      y = -abx + a²y - bc /  a² + b²
    '''
    Point2D.check_arg_point(A)
    Line2D.check_arg_line(l)
    a,b,c,x,y = l.a, l.b, l.c, A.x, A.y
    sqr_a,sqr_b = pow(a, 2), pow(b, 2)
    dnom = sqr_a + sqr_b
    nx = div((sqr_b * x) - (a * b * y) - (a * c), dnom)
    ny = div(-(a * b * x) + (sqr_a * y) - (b * c), dnom)
    return Point2D(nx, ny)

class Side2D:
  @staticmethod
  def check_arg_side(x):
    if isinstance(x, Side2D) is False:
      raise ArithmeticError("expected argument to be instance of class: Point2D")
    else:
      return True

  def __init__(self, A1, A2):
    '''
    A side A1 A2 ≡ {A1 , A2 } is a set with A1 and A2 points.
    The line of the side A1 A2 is the line A1 A2 .
    The side A1 A2 is a null side precisely when A1 A2 is a null line.
    '''
    Point2D.check_arg_point(A1)
    Point2D.check_arg_point(A2)
    if Point2D.is_distinct(A1, A2):
      self.line = Line2D.line_through_points(A1, A2)
      self.a = self.line.a
      self.b = self.line.b
      self.c = self.line.c
      self.p1 = A1
      self.p2 = A2
    else:
      raise ArithmeticError("constructor arguments must be distinct")
  def is_null(self):
    return self.line.is_null()

  def is_perpendicular_to(self, S):
    Side2D.check_arg_side(S)
    return self.line.is_perpendicular_to(S.line)

  def is_parallel_to(self, S):
    Side2D.check_arg_side(S)
    return self.line.is_parallel_to(S.line)

  def midpoint(self):
    '''
    For distinct points A1 ≡ [x1 , y1 ] and A2 ≡ [x2 , y2 ],
    the point [x1 + x2 / 2, y1 + y2 / 2] is the midpoint of the side A1 A2 .
    '''
    x1, x2, y1, y2 = self.p1.x, self.p2.x, self.p1.y, self.p2.y
    nx = div(x1 + x2, 2)
    ny = div(y1 + y2, 2)
    return Point2D(nx, ny)

  def Q(self):
    return Point2D.Q(self.p1,self.p2)

class Vertex2D:
  '''
  Definition A vertex l1 l2 ≡ {l1 , l2 } is a set with l1 and l2 intersecting lines.
  The point of the vertex l1 l2 is the point l1 l2 .
  The vertex l1 l2 is a null vertex precisely when l1 or l2 is a null line,
  and is a right vertex precisely when l1 and l2 are perpendicular.
  '''
  def __init__(self, l1, l2):
    Line2D.check_arg_line(l1)
    Line2D.check_arg_line(l2)
    if Line2D.is_parallel([l1, l2]) is False:
      self.point = Line2D.intersection_point(l1, l2)
      self.l1 = l1
      self.l2 = l2
      self.x = self.point.x
      self.y = self.point.y
    else:
      raise ArithmeticError("arguments to Vertex must two non-parallel lines")

  def is_null(self):
    return self.l1.is_null() or self.l2.is_null()

  def is_right(self):
    return Line2D.is_perpendicular(self.l1, self.l2)

class Triangle:
  '''
  A triangle is a set with A1,A2 and A3 points non-collinear points
  The points A1 , A2 and A3 are the points of the triangle A1 A2 A3 , and the lines
  l1 ≡ A2 A3 , l2 ≡ A1 A3 and l3 ≡ A1 A2 are the lines of the triangle. The sides A1 A2 ,
  A2 A3 and A1 A3 are the sides of the triangle, and the vertices l1 l2 , l2 l3 and l1 l3 are the vertices of the triangle.
  The point A1 is opposite the side A2 A3 , and so on, and the line l1 is opposite the
  vertex l2 l3 , and so on. With this terminology a triangle has exactly three points,
  necessarily non-collinear, and exactly three lines, necessarily non-concurrent. A triangle
  also has exactly three sides, and exactly three vertices.
  Definition A triangle A1 A2 A3 is a right triangle precisely when it has a right
  vertex. A triangle A1 A2 A3 is a null triangle precisely when one or more of its lines is a null line.
  In the rational or decimal number fields, there are no null triangles, since there are no null lines.
  '''
  @staticmethod
  def check_arg_side(x):
    if isinstance(x, Triangle) is False:
      raise ArithmeticError("argument is not an instance of Triangle")
    else:
      return True

  def __init__(self, A1, A2, A3):
    if Point2D.is_collinear([A1,A2,A3]) is False:
      self.A1 = A1
      self.A2 = A2
      self.A3 = A3
      self.L1 = Line2D.line_through_points(A2, A3)
      self.L2 = Line2D.line_through_points(A1, A3)
      self.L3 = Line2D.line_through_points(A1, A2)
      self.S1 = Side2D(A1, A2)
      self.S2 = Side2D(A2, A3)
      self.S3 = Side2D(A3, A1)
      self.V1 = Vertex2D(self.L1, self.L2)
      self.V2 = Vertex2D(self.L2, self.L3)
      self.V3 = Vertex2D(self.L3, self.L1)
      self.Q1 = Point2D.Q(self.A1, self.A2)
      self.Q2 = Point2D.Q(self.A2, self.A3)
      self.Q3 = Point2D.Q(self.A3, self.A1)
    else:
      raise ArithmeticError("triangle must be constructed with non-collinear points")

  def coordinate_values(self):
    x1,y1,x2,y2,x3,y3 = self.A1.x,self.A1.y,self.A2.x,self.A2.y,self.A3.x,self.A3.y
    return dict(x1=x1,y1=y1,x2=x2,y2=y2,x3=x3,y3=y3)

  def is_right(self):
    return self.V1.is_right() or self.V2.is_right() or self.V3.is_right()

  def is_null(self):
    return self.L1.is_null() or self.L2.is_null() or self.L3.is_null()


  def affine_centeroid(self):
    x1, x2, x3, y1, y2, y3 = self.A1.x, self.A2.x, self.A3.x, self.A1.y, self.A2.y, self.A3.y
    nx = div(x1 + x2 + x3, 2)
    ny = div(y1 + y2 + y3, 2)
    return Point2D(nx, ny)

  def circumquadrance(self):


  def circumcenter(self):
    x1, x2, x3, y1, y2, y3 = self.A1.x, self.A2.x, self.A3.x, self.A1.y, self.A2.y, self.A3.y
    dnom = 2 * comb.det_from_terms([x1, y2],d=3)
    cx = comb.det_from_terms([x1 ** 2, y2],d=3) + comb.det_from_terms([y1 ** 2, y1], d=3) / dnom
    cy = comb.det_from_terms([x1, x2 ** 2],d=3) + comb.det_from_terms([x1, y2 ** 2],d=3) / dnom
    return Point2D(cx, cy)


  def orthocenter(self):
    if self.is_null():
      return None
    else:
      x1, x2, x3, y1, y2, y3 = self.A1.x, self.A2.x, self.A3.x, self.A1.y, self.A2.y, self.A3.y
      dnom = comb.det_from_terms([x1, y2], 3)
      ox = comb.det_from_terms([x1, x2, y2], 3) + comb.det_from_terms([y1, y2 ** 2], 3) / dnom
      oy = comb.det_from_terms([x1, y1, y2], 3) + comb.det_from_terms([x1 ** 2, x2], 3) / dnom
      return Point2D(ox, oy)

  def incenter(self):





  def median(self):
    Line2D.
    '''
    A median of a triangle A1 A2 A3 is a line m
    passing through a point of the triangle and the midpoint of the opposite side.
    :return:
    '''



class Quadrilateral:
  '''
  A quadrilateral A1 A2 A3 A4 is a list [A1 , A2 , A3 , A4 ] of four distinct points,
  no three of which are collinear, with the conventions that
  A1 A2 A3 A4 ≡ A2 A3 A4 A1 and A1 A2 A3 A4 ≡ A4 A3 A2 A1

  The points A1 , A2 , A3 and A4 are the points of the quadrilateral
  and the lines l12 ≡ A1 A2 , l23 ≡ A2 A3 , l34 ≡ A3 A4 and l14 ≡ A1 A4 are the lines of the quadrilateral.
  The lines l13 ≡ A1 A3 and l24 ≡ A2 A4 are the diagonal lines (or just diagonals) of the quadrilateral.

  The sides A1 A2 , A2 A3 , A3 A4 and A1 A4 are the sides of the quadrilateral,
  and the vertices l12 l23 , l23 l34 , l34 l14 and l14 l12 are the vertices of the quadrilateral.
  The sides A1 A3 and A2 A4 are the diagonal sides of the quadrilateral.

  Two vertices which contain a common line are adjacent, otherwise vertices are opposite.
  Two sides which contain a common point are adjacent, otherwise sides are opposite.
  Note that the notation for quadrilaterals is different than for triangles.
  These notions generalize in an obvious way to defining n-gons A1 A2 · · · An ,
  except that only any three consecutive points are required to be non-collinear.

  A parallelogram is a quadrilateral A1 A2 A3 A4 with the property that both pairs of opposite sides are parallel,
  so that A1 A2 and A3 A4 are parallel, and A2 A3 and A1 A4 are parallel,
  A rectangle is a quadrilateral with the property that any pair of adjacent sides are perpendicular.
  Every rectangle is a parallelogram.
  '''
  def __init__(self, A1, A2, A3, A4):
    self.pset = set([A1, A2, A3, A4])
    if len(self.pset) != 4:
      raise ArithmeticError("arguments must be four distinct points")
    self.A1 = A1
    self.A2 = A2
    self.A3 = A3
    self.A4 = A4
    self.L12 = Line2D.line_through_points(self.A1, self.A2)
    self.L23 = Line2D.line_through_points(self.A2, self.A3)
    self.L34 = Line2D.line_through_points(self.A3, self.A4)
    self.L14 = Line2D.line_through_points(self.A1, self.A2)
    self.S12 = Side2D(self.A1, self.A2)
    self.S23 = Side2D(self.A2, self.A3)
    self.S34 = Side2D(self.A3, self.A4)
    self.S14 = Side2D(self.A1, self.A4)
    self.Diagonal1 = Side2D(self.A1, self.A3)
    self.Diagonal2 = Side2D(self.A2, self.A4)
    self.V1 = Vertex2D(self.L12, self.L23)
    self.V2 = Vertex2D(self.L23, self.L34)
    self.V3 = Vertex2D(self.L34, self.L14)
    self.V4 = Vertex2D(self.L14, self.L12)





'''
Theorem 9 (Aﬃne combination) Every point lying on the line A1 A2 is a unique
aﬃne combination λ1 A1 + λ2 A2 for some numbers λ1 and λ2 with λ1 + λ2 = 1, and
conversely any aﬃne combination of this form lies on A1 A2 .
'''














#quadrance between two 2d points
def QP_2d(p1, p2):
  return (abs(p2[0] - p1[0]) ** 2) + (abs(p2[1] - p1[1]) ** 2)













def same_shape(a,b):
  return np.shape(a) == np.shape(b)

def is_number_list(p):
  if isinstance(p, list) or isinstance(p, np.ndarray) and len(np.shape(p)) == 1:
    return array_check_all(p, lambda x: is_valid_number(x))
  else:
    return False

def slope2D(p1, p2):
  if is_number_list(p1) and is_number_list(p2) and same_shape(p1, p2):
    return Fraction(p2[1] - p1[1], p2[0] - p1[0])
  else:
    raise ArithmeticError("arguments p1 and p2 must be a list with integers or rationals and have the same length")


def mid_point(p1, p2):
  if is_number_list(p1) and is_number_list(p2) and same_shape(p1, p2):
    dim = len(p1)
    ret = []
    for i in range(dim):
      ret.append(Fraction(p1[i] + p2[i], 2))
    return np.array(ret)
  else:
    raise ArithmeticError("arguments p1 and p2 must be a list with integers or rationals and have the same length")
















