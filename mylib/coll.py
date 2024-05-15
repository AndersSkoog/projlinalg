from math import prod
def is_iterable(x):
  return hasattr(x,'__iter__')

def all_distinct(coll):
  if is_iterable(coll):
    return len(set(coll)) == len(coll)
  else:
    raise TypeError("call to function: distinct expected iterable, got:{t}".format(t=type(coll)))

def all(coll, pred):
  if is_iterable(coll):
    for el in coll:
      if pred(el) is False:
        return False
    return True
  else:
    raise TypeError("call to function: distinct expected iterable, got:{t}".format(t=type(coll)))

def any(coll, pred):
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






