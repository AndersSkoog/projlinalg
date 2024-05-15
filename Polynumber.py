import math
import numpy as np
from mylib.coll import add_indices2, mul_indices2

class Polynumber:

  def __init__(self, arr):
    self.arr = arr

  def tail_zeroes(self):
    _arr = self.arr
    ret = []
    while _arr[-1] == 0:
      ret.append(_arr.pop())
    return ret

  def no_zero_arr(self):
    zl = self.tail_zeroes()
    return self.arr[0:len(self.arr) - len(zl)]

  def __eq__(self, other):
    if isinstance(other, Polynumber):
      return self.no_zero_arr() == other.no_zero_arr()
    else:
      return False

  def __add__(self, other):
    if isinstance(other, Polynumber):
      len1 = len(self.arr)
      len2 = len(other.arr)
      if len1 == len2:
        narr = add_indices2(self.arr, other.arr)
        return Polynumber(narr)
      else:
        rest = self.arr[len2:len1] if len1 > len2 else other.arr[len2:len1]
        len_min = min(len1,len2)
        a1 = self.arr[0:len_min]
        a2 = other.arr[0:len_min]
        narr = add_indices2(a1, a2)
        return narr + rest
    else:
      raise ArithmeticError("can only be added with a polynumber")

  def __mul__(self, other):
    '''
    0 = 0 + 0
    1 = 0 + 1, 1 + 0
    2 = 0 + 1 + 1 + 0, 1 + 1, 2 + 0, 0 + 2
    3 = 0 + 3, 3 + 0, 2 + 1, 1 + 2
    4 = 0 + 4, 4 + 0, 2 + 2, 1 + 1,
    :param other:
    :return:
    '''



  def scale(self, num):
    ret = []
    for el in self.arr:
      ret.append(num * el)
    return Polynumber(ret)

