from fractions import Fraction
import math

def binomial(m, k):
 nom = math.factorial(m + k)
 dnom = math.factorial(m) * math.factorial(k)
 return nom / dnom

def trinomial(m, k, l):
  nom = math.factorial(m + k + l)
  dnom = math.factorial(m) * math.factorial(k) * math.factorial(l)
  return nom / dnom

def pyth_tripple(m, n):
  m2, n2 = m**2, n**2
  a = (m2 - n2) ** 2
  b = (2*m*n) ** 2
  c = (m2 + n2) ** 2
  return [[a, b, c], a+b, c]


def sieve(n):
    numbers = [True] * n
    primes = []
    n_sqrt = pow(n, 0.5)
    i1 = 2
    while i1 < n_sqrt:
        i1 += 1
        j = i1 * i1
        # print(j)
        while j < n:
            # print(j)
            numbers[j] = False
            j += i1

    for i in range(2, n):
        if numbers[i]:
            primes.append(i)
    return primes

first_primes = sieve(1000)

def incMixed(tup, bases):
  result = list(map(lambda v: v, tup))
  result[0]+=1
  for k in range(len(tup)):
    if result[k] <= bases[k]:
      break
    elif k != (len(tup) - 1):
      result[k] = 0
      result[k+1]+=1
    else:
      result[k] = 0
  return result

def factors(num):
    ret = []
    for n in range(1, num):
      if num % n == 0:
        ret.append(n)
    return ret


def factor(n):
  primes = first_primes if first_primes[-1] < n else sieve(n)
  factors = []
  _n: int = n
  for k in range(len(primes)):
    p: int = primes[k]
    if (_n % p) == 0:
      f = dict(prime=p,power=0)
      while _n % p == 0:
        f["power"] += 1
        _n /= p
      factors.append(f)

  if _n > 1:
    factors.append(dict(prime=_n, power=1))
  return factors

def divisors(n):
  factors = factor(n)
  powers = [0] * len(factors)
  max_powers = list(map(lambda _dict: _dict["power"], factors))
  ret = [1]
  while True:
    powers = incMixed(powers, max_powers)
    d = math.prod([pow(factors[i]["prime"], powers[i]) for i in range(len(powers))])
    if d == 1:
      break
    ret.append(d)

  ret.sort()
  return ret


def div(nom, dnom):
  if nom // dnom - nom / dnom == 0:
    return nom // dnom
  else:
    return Fraction(nom, dnom)


def is_even(v):
    return v % 2 == 0

def is_odd(v):
    return v % 2 == 1


def quotient(a, b):
  r = a % b
  return (a - r) // b


def is_prime(x):
  if isinstance(x, int):
    return x in first_primes
  else:
    raise ArithmeticError("x must be integer")

def gcd(a, b):
  _a = a if a > 0 else -a
  _b = b if b > 0 else -b
  while True:
    if _b == 0:
        return _a
    _a %= _b
    if a == 0:
        return _b
    _b %= _a

def lcm(a, b):
  return a * b / gcd(a, b)


def is_square(n):
  return math.sqrt(n) % 1 == 0

def harm_mean(A, B):
  nom = 2 * (A*B)
  dnom = A + B
  return nom / dnom

def arithmetic_mean(A, B):
  return (A + B) / 2

def gnomonic_number(n):
    return 1 if n == 1 else pow(n, 2) - pow(n - 1, 2)


def polygon_num(s, n):
    t1 = (n * (n - 1)) / 2
    t2 = s - 2
    res = t2 * t1 + n
    return 1 if n <= 1 else int(res)

def centered_polygon_num(s,n):
    return 1 if n <= 1 else int((s * pow(n,2) - s * n + 2) / 2)


def tetrahedral_numbers(n):
    return 1 if n <= 1 else int((n * ((n + 1) * (n + 2))) / 6)

def cubic_number(n):
    return 1 if n <= 1 else pow(n, 3)

def octahedral_number(n):
    return 1 if n <= 1 else int((n * (pow(2 * n, 2) + 1)) / 3)

def icosahedral_number(n):
     n2 = pow(n, 2)
     n5 = 5 * n
     return 1 if n <= 1 else int((n * (5*n2 - n5 + 2)) / 2)

def dodecahedral_number(n):
    nom = n * ((3*n) - 1) * ((3*n) - 2)
    return 1 if n <= 1 else int(nom / 2)

def stella_octangula_number(n):
    return 1 if n <= 1 else octahedral_number(n) + (8 * tetrahedral_numbers(n - 1))

def rhombic_dodecahedron_number(n):
    return 1 if n <= 1 else centered_cube_number(n) + (6 * m_gon_pyramid(4, n - 1))

def centered_cube_number(n):
    return 1 if n <= 1 else pow(n, 3) + pow(n - 1, 3)

def m_gon_pyramid(m, n):
    nom = n * (n + 1) * ((m - 2) * n - m + 5)
    dnom = 6
    return 1 if n <= 1 else int(nom / dnom)

def centered_mgon_pyramid(m, n):
    n2 = 2 * n - 1
    m_1 = m - 1
    nn = pow(n,2)
    nom = (n2 * m_1 * nn) - (m_1 * n + 6)
    return 1 if n <= 1 else int(nom/2)
    #return 1 if n <= 1 else int(((2*n - 1) * (s - 1) * pow(n,2) - (s - 1) * n + 6) / 2)

def seq(s_start,s_stop,init,fn):
    ret = init
    for i in range(s_start,s_stop):
        nv = fn(ret,i)
        ret.append(nv)
    return ret

def get_perfect(doubles_seq):
    s = sum(doubles_seq)
    last = doubles_seq[len(doubles_seq) - 1]
    return s * last


def polygon_diagonals(sides):
    return sides * (sides - 3) / 2


def algo_seq(n_start, n_stop, fn, init):
    ret = init
    for i in range(n_start, n_stop):
        v = fn(ret, i)
        if v:
         ret.append(v)
    return ret


def solid_numbers(n_iter):
    laterals_a = []
    laterals_b = []
    diagonals_a = []
    diagonals_b = []
    prev_s = 1
    prev_d = 1
    for i in range(1,n_iter):
      s = prev_s + prev_d
      d = prev_d + (prev_s * 2)
      s2 = s ** 2
      d2 = d ** 2
      prev_s = s
      prev_d = d
      laterals_a.append(s)
      laterals_b.append(s2)
      diagonals_a.append(d)
      diagonals_b.append(d2)
    return [laterals_a,laterals_b,diagonals_a,diagonals_b]


solids = solid_numbers(100)
laterals_a = solids[0]
laterals_b = solids[1]
diagonals_a = solids[2]
diagonals_b = solids[3]
primes = sieve(100)
naturals = list(range(1,102))
doubles = seq(0,40,[1],lambda arr,i: arr[i] * 2)
evens = list(range(2,202,2))
odds = list(range(1,202,2))
oblongs = list(range(2,200,2))
squares = list(map(lambda v: v * v, naturals))
unequalaterals = [evens[i - 1] + evens[i] for i in range(1,len(evens) - 1)]
perfects = [get_perfect(doubles[0:i]) for i in range(1,19)]
geometricmeans = [int(pow(squares[i - 1] * squares[i],0.5)) for i in range(1,len(naturals))]
abundants = algo_seq(12, 100, lambda arr,i: i if sum(factor(i)) > i else False,[])
deficients = algo_seq(1,100, lambda arr,i: i if sum(factor(i)) < i else False, [])
successive_factors = [factor(i) for i in range(0,100)]
triangulars = [sum(range(1,v)) for v in range(2,100)]
squares2 = [polygon_num(4,n) for n in range(1,100)]
pentagons = [polygon_num(5, n) for n in range(1,100)]
hexagons = [polygon_num(6,n) for n in range(1,100)]
heptagons = [polygon_num(7,n) for n in range(1,100)]



