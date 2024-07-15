import math


def rotor_coord(r, h):
  x = r * C(h)
  y = r * S(h)
  v = [x, y]
  h_of_v = (r - x) / y
  unit_vector = [x / r, y / r]
  
  length_of_v = math.sqrt(pow(x, 2) + pow(y, 2))



def rotor(h):
  l = line_through_points([-1, 0], [0, h])
  v = [C(h), S(h)]
  r = math.sqrt(pow(v[0], 2) + pow(v[1], 2))



def C(h):
    (1 - pow(h, 2)) / (1 + pow(h, 2))

def S(h):
  return (2 * h) / (1 + pow(h, 2))

def T(h):
  return S(h) / C(h)

def M(h):
  ret = 2 / (1 + pow(h, 2))
  print(ret, 1 + C(h), S(h) / h)
  return ret
def line_through_points(p1, p2):
  x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
  a = y1 - y2
  b = x2 - x1
  c = (x1*y2) - (x2*y1)
  return [a, b, c]

def quadramce(v):
  x, y = v[0], v[1]
  return pow(x, 2) + pow(y, 2)

def spread(v1, v2):
  x1, y1, x2, y2 = v1[0], v1[1], v2[0], v2[1]
  nom = (x1*y2) - (x2*y1)
  dnom = (pow(x1, 2) + pow(y1, 2)) * (pow(x2, 2) + pow(y2, 2))
  return nom / dnom

def point_on_unit_circle(t):
  if t != -1:
    t_sqr = pow(t, 2)
    x = (1 - t_sqr) / (1 + t_sqr)
    y = (2 * t) / (1 + t_sqr)
    return [x, y]
  else:
    raise ArithmeticError("t can not be -1")

def derC(h):
  return -S(h) * M(h)

def derS(h):
  return C(h) * S(h)


