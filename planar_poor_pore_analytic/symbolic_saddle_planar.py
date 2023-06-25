#%%
import sympy as sp
from sympy import symbols
import numpy as np

x, y = symbols("x y")
a, b, V, s, r, c = symbols("a b V s r c", real = True)
#y = a*x**2 + b*x + s
y = a*x**2 + b*x + s
#y = a*x**3 + b*x**2 + c*x**1 + s
#yp = y.diff(x)
#y = s + sp.sqrt(a*x**2 + b*x)
V_ = sp.integrate(y, (x, 0, r))
# %%
#V_ = r*s +V_.args[1].args[0].args[0]
# %%
res = sp.solve([sp.Eq(y.subs(x, r), 0), sp.Eq(V, V_)], [a, b])
#res = sp.solve([sp.Eq(y.subs(x, r), 0), sp.Eq(V, V_), sp.Eq(yp.subs(x, r), 0)], [a, b, c])
a_ = res[a]
b_ = res[b]
#c_ = res[c]
#%%
y_ = y.subs({a:a_, b:b_}).expand().collect(x)
#y_ = y.subs({a:a_, b:b_, c:c_}).expand().collect(x)
y_
# %%
def Y(s_, r_, V_):
    #return sp.lambdify(x, y_.subs({s:s_, r:r_, V:V_}))
    return sp.lambdify(x, y_.subs({s:s_, r:r_, V:V_}))

def S(s_, r_, V_, scipy = True):
    dS = sp.sqrt(1+y_.subs({s:s_, r:r_, V:V_}).diff(x)**2)
    if scipy:
        from scipy.integrate import quad
        dS_func = sp.lambdify(x, dS)
        return quad(dS_func, 0, r_)[0]
    return sp.N(sp.integrate(dS, (x, 0, r_)))
# %%
S(1,1,1)
# %%
