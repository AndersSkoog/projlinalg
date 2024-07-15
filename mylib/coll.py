from math import prod
def is_iterable(x):
  return hasattr(x, '__iter__')

def all_distinct(coll):
  if is_iterable(coll):
    return len(set(coll)) == len(coll)
  else:
    raise TypeError("call to function: distinct expected iterable, got:{t}".format(t=type(coll)))

def all_pass_pred(coll, pred):
  if is_iterable(coll):
    for el in coll:
      if pred(el) is False:
        return False
    return True
  else:
    raise TypeError("call to function: distinct expected iterable, got:{t}".format(t=type(coll)))

def any_pass_pred(coll, pred):
  if is_iterable(coll):
    for el in coll:
      if pred(el):
        return True
    return False
  else:
    raise TypeError("call to function: distinct expected iterable, got:{t}".format(t=type(coll)))

def select(coll, where):
  if is_iterable(coll):
    ret = []
    for el in coll:
      if where(el):
        ret.append(el)
    return ret
  else:
    raise TypeError("call to function: distinct expected iterable, got:{t}".format(t=type(coll)))

def first(coll, where):
  if is_iterable(coll):
    ret = None
    for el in coll:
      if where(el):
        ret = el
        break
    return ret
  else:
    raise TypeError("call to function: distinct expected iterable, got:{t}".format(t=type(coll)))

def add_indices2(a1, a2):
  return list(map(lambda t: sum([t[0], t[1]]), zip(a1, a2)))

def add_indices3(a1, a2, a3):
  return list(map(lambda t: sum([t[0], t[1], t[2]]), zip(a1, a2, a3)))

def add_indices4(a1, a2, a3):
  return list(map(lambda t: sum([t[0], t[1], t[2], t[3]]), zip(a1, a2, a3)))

def mul_indices2(a1, a2):
  return list(map(lambda t: prod([t[0], t[1]]), zip(a1, a2)))

def mul_indices3(a1, a2, a3):
  return list(map(lambda t: prod([t[0], t[1], t[2]]), zip(a1, a2, a3)))

def mul_indices4(a1, a2, a3, a4):
  return list(map(lambda t: prod([t[0], t[1], t[2], t[3]]), zip(a1, a2, a3, a4)))

def find_min(arr, ind):
  """
  given a list of lists with numberic entries
  returns the element with the highest value at index ind
  """
  ind_vals = list(map(lambda p: p[ind], arr))
  ind = ind_vals.index(min(ind_vals))
  return arr[ind]

def find_max(arr, ind):
  """
  given a list of lists with numberic entries
  returns the element with the highest value at index ind
  """
  ind_vals = list(map(lambda p: p[ind], arr))
  ind = ind_vals.index(max(ind_vals))
  return arr[ind]

def find_with_pred(arr, pred):
  for el in arr:
    if pred(el):
      return el
  return None






