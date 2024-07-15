
#shorthand for creating a square matrix from a one dimensional list of arbitary length
def fill_sqr_mtx(dim, fill):
  ret = [[0] * dim for _ in range(dim)]
  for i in range(dim):
    for j in range(dim):
      ind = ((dim * i) + j) % len(fill)
      ret[i][j] = fill[ind]
  return ret
