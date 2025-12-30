### Geomatica
... is a library for implementing Geometric Algebra in Python


After `pip install geomatica`, you can do all of this:

```python
from geomatica import GA

ga = GA() # create a Geometric Algebra

i,j,k = ga[1:4] # extract 3 basis vectors

v = 3*i-5*j+k # some vector
S = (i+j-k)^(-j+2*k) # some bivector, constructed using wedge product

print(-i*j*k*S) # hodge dual of bivector S in R3
print((v|S)/S) # project v onto S through dot product + multiplication by inverse of S
```

Further information can be found in docstrings.

---
pypi: [HERE!](https://pypi.org/project/geomatica)

github: [HERE!](https://github.com/svntythsnd/geomatica)