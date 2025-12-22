from typing import Callable as _Callable, Protocol as _Protocol, Union as _Union
class IMultivector(_Protocol):
 def __add__(self, other: _Union[int, float, 'IMultivector']) -> 'IMultivector':
  pass
 def __radd__(self, other: _Union[int, float, 'IMultivector']) -> 'IMultivector':
  pass
 def __sub__(self, other: _Union[int, float, 'IMultivector']) -> 'IMultivector':
  pass
 def __rsub__(self, other: _Union[int, float, 'IMultivector']) -> 'IMultivector':
  pass
 def __neg__(self) -> 'IMultivector':
  pass
 def __invert__(self) -> 'IMultivector':
  """Return the adjugate of the Multivector.""" 
 def __mul__(self, other: _Union[int, float, 'IMultivector']) -> 'IMultivector':
  """Compute e^M either by decomposing the Multivector into commuting blocks or, if that fails, explicit Taylor expansion.""" 
 def __rmul__(self, other: int | float) -> 'IMultivector':
  pass
 def __pow__(self, other: int | float) -> 'IMultivector':
  pass
 def __rpow__(self, other: int | float) -> 'IMultivector':
  """Exponentiate the Multivector by applying e^(M ln b).""" 
 def __abs__(self) -> float:
  """Return the determinant of the Multivector.""" 
 def __matmul__(self, grade: int) -> 'IMultivector':
  """Extract a specific grade of the Multivector.""" 
 def __truediv__(self, other: _Union[int, float, 'IMultivector']) -> 'IMultivector':
  pass
 def __rtruediv__(self, other: int | float) -> 'IMultivector':
  pass
 def __or__(self, other: _Union[int, float, 'IMultivector']) -> 'IMultivector':
  """Return the dot product of two Multivectors.""" 
 def __xor__(self, other: _Union[int, float, 'IMultivector']) -> 'IMultivector':
  """Return the wedge product of two Multivectors.""" 
 def __pos__(self) -> int | None:
  """Return the grade of the multivector if it's a blade, None otherwise.""" 
 def __format__(self, form: str) -> str:
  pass
 def __str__(self) -> str:
  pass
 def exp(self) -> 'IMultivector':
  pass
 
_subscripts = str.maketrans('0123456789','₀₁₂₃₄₅₆₇₈₉')
def _merge_sort_parity(arr):
 if len(arr) <= 1 : return arr, 1
 mid = len(arr) // 2
 left, p_left = _merge_sort_parity(arr[:mid])
 right, p_right = _merge_sort_parity(arr[mid:])
 merged = []
 parity = p_left * p_right
 i = j = 0
 while i < len(left) and j < len(right):
  if left[i] <= right[j]:
   merged.append(left[i])
   i += 1
   continue
  merged.append(right[j])
  j += 1
  if (len(left) - i) & 1: parity = -parity
 merged.extend(left[i:])
 merged.extend(right[j:])
 return merged, parity
class GA:
 def __init__(ga, *, signature:_Callable[[int], float]= lambda x:1.0, epsilon_order:int=0):
  """
        Create a Geometric Algebra.

        Args:
            signature: a Callable returning the square of the nth basis vector.
                       Defaults to 1 for all.
            epsilon_order:  integer offset for machine epsilon comparisons.
                       The effective bound for treating numbers as zero is
                       2^-epsilon_order times the machine epsilon. Defaults to 0.
        """
  ga.signature = signature
  ga.epsilon_order = epsilon_order
  class Multivector:
   def __init__(self, keys:dict[int, float], **argv) -> None:
    from math import ldexp
    self.__d = {k:v for k, v in keys.items() if 1+abs(ldexp(v,-ga.epsilon_order)) != 1}
    self.__decomposable = argv.get("decomposable",None)
    self.__decomposition = argv.get("decomposition",None)
    self.__sigma = None
    
   def __add__(self, other: _Union[int, float, 'Multivector']) -> 'Multivector':
    if isinstance(other, int | float) : return Multivector({0: self.__d.get(0, 0) + other,**{mask: value for mask, value in self.__d.items() if mask != 0}},decomposable=self.__decomposable, decomposition=self.__decomposition)
    if not isinstance(other, Multivector) : return NotImplemented
    return Multivector({mask: self.__d.get(mask, 0) + other._Multivector__d.get(mask, 0) for mask in sorted(self.__d.keys() | other._Multivector__d.keys())})
   def __radd__(self, other: _Union[int, float, 'Multivector']) -> 'Multivector' : return self+other
   def __sub__(self, other: _Union[int, float, 'Multivector']) -> 'Multivector' : return self+(-other)
   def __rsub__(self,other: _Union[int, float, 'Multivector']) -> 'Multivector': return -self+other
   def __neg__(self) -> 'Multivector' : return Multivector({mask: -val for mask, val in self.__d.items()}, decomposition=self.__decomposition, decomposable=self.__decomposable)
   def __invert__(self) -> 'Multivector':
    sigma = self.__get_sigma()
    return Multivector({k: -v if (sigma >> n)&1 else v for n, (k, v) in enumerate(self.__d.items())},decomposition=self.__decomposition, decomposable=self.__decomposable)
   def __rmul__(self, other: int | float) -> 'Multivector':
    if isinstance(other, int | float) : return self*other
    return NotImplemented
   def __pow__(self, other: int | float) -> 'Multivector':
    if not isinstance(other, int | float) : return NotImplemented
    from math import ldexp
    if 1 + abs(ldexp(other, -ga.epsilon_order)) % 1 != 1: raise ValueError(f'Multivector exponent must be an integer, but got {other}')
    other -= other % 1
    if other == 0 : return ga[0]
    if other < 0: out = (~self)/abs(self)
    else: out = self
    for _ in range(abs(other)-1): out *= self
    return out
   def __abs__(self) -> float : return (self*~self)._Multivector__d.get(0, 0)
   def __get_sigma(self):
    if self.__sigma is not None : return self.__sigma
    blades = list(self.__d.keys())
    if added := blades[0] != 0: blades = [0] + blades
    from collections import deque
    n = len(blades)
    grades = [b.bit_count() for b in blades]
    epsilon = [[0]*n for _ in range(n)]
    for i in range(n):
     for j in range(n):
      t = (blades[i] & blades[j]).bit_count()
      parity = (grades[i]*grades[j] - t) & 1
      epsilon[i][j] = -1 if parity else +1
     
    sigma = 0
    known = 1
    queue = deque([0])
    while queue:
     i = queue.popleft()
     si = -1 if (sigma >> i) & 1 else +1
     for j in range(n):
      if i == j: continue
      required_sign = -epsilon[i][j] * si
      required_bit = 1 if required_sign == -1 else 0
      mask = 1 << j
      if not (known & mask):
       if required_bit: sigma |= mask
       else: sigma &= ~mask
       known |= mask
       queue.append(j)
       continue
      current_bit = (sigma >> j) & 1
      if current_bit != required_bit: raise ValueError("No adjugate exists (inconsistent constraints)")
     
    return sigma >> 1 if added else sigma
   def __rpow__(self, other: int | float) -> 'Multivector':
    if not isinstance(other, int | float) : return NotImplemented
    import math
    return (math.log(other)*self).exp()
   @staticmethod
   def __mulbases(mask1, mask2):
    val = 1
    bases = [i for i in range(mask1.bit_length()) if (mask1 >> i) & 1] + [i for i in range(mask2.bit_length()) if (mask2 >> i) & 1]
    seen = set()
    for basis in tuple(bases):
     if basis in seen: continue
     seen.add(basis)
     diff = 0
     keep = False
     for n, factor in enumerate(reversed(tuple(bases))):
      if factor != basis: continue
      keep = not keep
      if keep:
       diff = n
       continue
       
      if n % 2 == diff % 2: val *= -1
      bases.pop(~diff)
      bases.pop(~n+1 if n>diff else ~n)
      val *= ga.signature(basis)
      
     
    bases, parity = _merge_sort_parity(bases)
    bases = sum(1 << i for i in bases)
    return bases, val*parity
   def __or__(self, other: _Union[int, float, 'Multivector']) -> 'Multivector':
    if isinstance(other, int | float) : return self*other
    new = {}
    for mask1, val1 in self.__d.items():
     for mask2, val2 in other._Multivector__d.items():
      if (mask1^mask2).bit_count() != abs(mask1.bit_count() - mask2.bit_count()): continue
      mask, basisprod = self.__mulbases(mask1, mask2)
      new[mask] = new.get(mask, 0) + val1*val2*basisprod
     
    return Multivector(dict(sorted(new.items())))
   def __xor__(self, other: _Union[int, float, 'Multivector']) -> 'Multivector':
    if isinstance(other, int | float) : return ga[-1]
    new = {}
    for mask1, val1 in self.__d.items():
     for mask2, val2 in other._Multivector__d.items():
      if (mask1^mask2).bit_count() != mask1.bit_count() + mask2.bit_count(): continue
      mask, basisprod = self.__mulbases(mask1, mask2)
      new[mask] = new.get(mask, 0) + val1*val2*basisprod
     
    return Multivector(dict(sorted(new.items())))
   def __mul__(self, other: _Union[int, float, 'Multivector']) -> 'Multivector':
    from math import ldexp
    if isinstance(other, int | float) : return Multivector({mask: other*val for mask, val in self.__d.items()},decomposable=self.__decomposable, decomposition=self.__decomposition) if 1+abs(ldexp(other, -ga.epsilon_order)) != 1 else Multivector({})
    elif not isinstance(other, Multivector) : return NotImplemented
    new = {}
    for mask1, val1 in self.__d.items():
     for mask2, val2 in other._Multivector__d.items():
      mask, basisprod = self.__mulbases(mask1, mask2)
      new[mask] = new.get(mask, 0) + val1*val2*basisprod
     
    return Multivector(dict(sorted(new.items())))
   def __matmul__(self, grade: int) -> 'Multivector':
    if not isinstance(grade, int) : return NotImplemented
    return Multivector({mask: val for mask, val in self.__d.items() if mask.bit_count() == grade})
   def __decompose(self):
    if self.__decomposable is not None : return self.__decomposition
    commutes = lambda mask1,mask2: mask1.bit_count()*mask2.bit_count()%2 == (mask1&mask2).bit_count()%2
    blocks = []
    for mask in self.__d.keys():
     noncom = 0
     noncomindex = None
     for n, block in enumerate(blocks):
      comtrack = None
      for item in block:
       if comtrack is None:
        comtrack = commutes(mask, item)
        continue
       if comtrack is not commutes(mask, item):
        self.__decomposable = False
        self.__decomposition = None
        return None
       
      if comtrack is False:
       noncomindex = n
       noncom+=1
      
     if noncom == 0: blocks.append([mask])
     elif noncom == 1: blocks[noncomindex].append(mask)
     else:
      self.__decomposable = False
      self.__decomposition = None
      return None
     
    self.__decomposable = True
    self.__decomposition = blocks
    return blocks
   def exp(self) -> 'Multivector':
    d = self.__decompose()
    if not self.__decomposable:
     from math import ldexp
     current = self
     cumulus = ga[0] + current
     n = 2
     while True:
      current *= self
      current /= n
      if 1+abs(ldexp(sum(s*s for s in current.__d.values()),-ga.epsilon_order)) == 1: break
      cumulus += current
      n+=1
     return sum
    import math
    prod = 1.0
    for block in d:
     if block == [0]:
      prod *= math.exp(self.__d[0])
      continue
     sum = 0
     for mask in block:
      norm = -1 if 2 <= mask.bit_count() % 4 <= 3 else 1
      for i in range(mask.bit_length()): norm *= ga.signature((mask >> i) & 1)
      sum += norm * self.__d[mask]**2
     value = math.sqrt(abs(sum))
     prod *= Multivector({0: math.cosh(value), **{mask:math.sinh(value)*self.__d[mask]/value for mask in block}})if sum > 0else Multivector({0: 1, **{mask: self.__d[mask] for mask in block}})if sum == 0else Multivector({0: math.cos(value), **{mask:math.sin(value)*self.__d[mask]/value for mask in block}})
    return prod
   def __pos__(self) -> int | None:
    grade = None
    for mask in self.__d.keys():
     if grade is None: grade = mask.bit_count()
     elif mask.bit_count() != grade : return None
    return grade
   def __truediv__(self, other: _Union[int, float, 'Multivector']) -> 'Multivector':
    if isinstance(other, int | float) : return Multivector({mask: value/other for mask, value in self.__d.items()})
    if not isinstance(other, Multivector) : return NotImplemented
    return self*(other**(-1))
   def __rtruediv__(self, other: int | float) -> 'Multivector':
    if not isinstance(other, int | float) : return NotImplemented
    return other*(self**(-1))
   def __format__(self, form: str) -> str : return ''.join(('+' if value > 0 else '') + format(value, form) + ''.join('e' + str(i+1).translate(_subscripts) for i in range(mask.bit_length()) if (mask >> i) & 1)for mask, value in self.__d.items()).removeprefix('+') if self.__d else format(0.0,form)
   def __str__(self) -> str : return f'{self:g}'
  ga.Multivector = Multivector
 def __getitem__(self, n: int) -> IMultivector:
  """
        Get the nth basis vector of the GA if n > 1, the unit scalar if n = 0
        and the zero Multivector if n < 0.
        """
  return self.Multivector({(1<<(n-1) if n > 0 else 0): 1.0} if n >= 0 else {})
 

