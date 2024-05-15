from itertools import permutations
from typing import List, Any
import numpy as np

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
  return int(round(np.linalg.det(det_matrix(terms))))


