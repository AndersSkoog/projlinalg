import itertools
import numpy as np
import math


def det(m):
  dim = len(m)
  t1 = []
  t2 = []
  for it1 in range(dim):
    t1.append(math.prod([m[j1][(it1 + j1) % dim] for j1 in range(dim)]))
  for it2 in list(reversed(range(dim))):
    t2.append(math.prod([m[j2][(it2 - j2) % dim] for j2 in range(dim)]))
  return sum(t1) - sum(t2)


def det_matrix(terms, d):
  if d > len(terms):
    ones = [1] * (d - len(terms))
    _terms = terms + ones
    pl = list(itertools.permutations(_terms))
    ret = []
    for i in range(d):
      row = [pl[j][i] for j in range(len(terms))] + ones
      ret.append(row)
    print(ret)
    return np.matrix(ret)
  elif d == len(terms):
    pl = list(itertools.permutations(terms))
    print(pl, terms)
    ret = []
    for i in range(d):
        row = [pl[j][i] for j in range(d)]
        print(row)
        ret.append(row)
    print(ret)
    return np.matrix(ret)

  else:
    raise ArithmeticError("d must be >= length of argument terms array")


def det_from_terms(terms, d=3):
  return int(round(np.linalg.det(det_matrix(terms, d))))


