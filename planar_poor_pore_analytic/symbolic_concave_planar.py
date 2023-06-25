# %%
from IPython.display import Markdown as md
from IPython.display import SVG
import sympy as sp
from sympy import symbols

"""
x = symbols("x", positive = True)
ypp = symbols("y^{\prime\prime}", negative = True)
yp = symbols("y^{\prime}", negative = True)
y = symbols("y", positive = True)
A = symbols("A", positive =True)
C= symbols("C")
l = symbols("\lambda", positive = True)
s = symbols("s", positive = True)
V = symbols("V")
r = symbols("r")
"""
x = symbols("x")
ypp = symbols("y^{\prime\prime}")
yp = symbols("y^{\prime}")
y = symbols("y")
A = symbols("A")
C= symbols("C")
l = symbols("\lambda")
s = symbols("s")
V = symbols("V")
r = symbols("r")

# %%
SVG(filename ="drawing.svg")

# %% [markdown]
# Minimize surface
# $$
# dS = \sqrt{1+y^{\prime}} dx
# $$
# with constant volume
# $$
# dV = y dx
# $$
# Functional to minimize
# $$
# F = S + \lambda V
# $$
# 
# other constrains:
# $$
# y(0) = s/2
# \\
# y(r) > 0
# \\
# y^{\prime}(r) = 0
# \\
# y^{\prime\prime}(x) > 0 \text{ if } y(r)<y(0)
# \\
# y^{\prime\prime}(x) \le 0 \text{ otherwise}
# $$
# 
# $$
# F(y^{\prime}, y) = \sqrt{1+(y^{\prime})^2} + \lambda y
# $$
# 
# Beltrami identity
# 
# $$
# F - y^{\prime} \frac{\partial F}{\partial y^{\prime}} = C
# $$

# %%
F = sp.sqrt(1+yp**2) + l*y
second = F.diff(yp)*yp
second_term = r"y^{\prime} \frac{\partial F}{\partial y^{\prime}}"
md(f"$${second_term} = {sp.latex(second)}$$")

# %%
lhs = F - second - C
display(lhs)
lhs = lhs.simplify()
eq = sp.Eq(lhs, 0)
display(eq)


# %%
#display(sp.Eq(symbols("d/dx")*lhs, 0))
#lhs_diff = lhs.diff(y)
print("let x = r")
yr, ypr, yppr = symbols(r"y_r y^{\prime}_r y^{\prime\prime}_r")
eq_r = eq.subs(y, yr).subs(yp, ypr)
display(eq_r)
eq_r = eq_r.subs(ypr, 0)
C_ = sp.solve(eq_r, C)[0]
display(sp.Eq(C, C_))
eq2 = sp.Eq(lhs.subs(C,C_), 0)
display(eq2)

#print("let x = 0")
#eq2 = eq2.subs(y, s)
#display(eq2)
#l_ = sp.solve(eq2, l)[0].simplify()
#display(sp.Eq(l, l_))


# %%
#lhs_diff = eq2.args[0].diff(y)*yp + eq2.args[0].diff(yp)*ypp
#display(sp.Eq(lhs_diff, 0))
#lhs_diff = (lhs_diff/yp).simplify()
#display(sp.Eq(lhs_diff, 0))
#print("let x = 0")
#eq3 = sp.Eq(lhs_diff.subs(yp, 0).subs(ypp, yppr), 0)
#display(eq3)
#l_ = sp.solve(eq3, l)[0]
#display(sp.Eq(l, l_))

# %%
lhs_ = lhs.subs(C, C_)#.subs(l, l_)
display(sp.Eq(lhs_, 0))

# %%
yp_roots = sp.solve(lhs_, yp)
display(*[sp.Eq(yp, yp_root) for yp_root in yp_roots])

# %%
dx, dy = symbols("dx dy")
pm = symbols("\pm1")
yp_ = yp_roots[1].simplify()*pm
lhs = sp.solve((yp_ - yp).subs(yp, dy/dx), dx)[0]
sp.Eq(lhs, dx)

# %%

u = 1 - (l*(y-yr)-1)**2
print("substitute")
display(sp.Eq(symbols("u"), u))

du = u.diff(y)
eq=sp.Eq(symbols("du"), du*dy)
display(eq)
dy_ = sp.solve(eq, dy)[0]
display(sp.Eq(symbols("dy"), dy_))

# %%
u_ = symbols("u")
rhs = x-A
integrand = pm*1/(2*l*sp.sqrt(u_))
print("integrand:")
display(integrand)
print("after integration")
lhs = sp.integrate(integrand, u_)
eq = sp.Eq(lhs, rhs)
display(eq)
print("finally, after substitution")
eq = eq.subs(u_, u)
display(eq)

# %%
eq2 = sp.Eq(eq.args[0]**2, eq.args[1]**2).subs(pm, 1)
display(eq2)
print("let x = r")
eqA = eq2.subs(x, r).subs(y, yr).simplify()
display(eqA)
A_ = sp.solve(eqA , A)[0]
display(sp.Eq(A, A_))
eq2 = eq2.subs(A, A_)
display(eq2)
print("let x = 0")
eq0 = eq2.subs(x, 0).subs(y, s)
display(eq0)

# %%
l_ = sp.solve(eq0, l)[0].simplify()
display(sp.Eq(l, l_))
eq2 = eq2.subs(l, l_).simplify()
eq2 = sp.Eq(eq2.args[0], eq2.args[1].factor())
display(eq2)

# %%
y_roots = sp.solve(eq2 , y)
display(*[sp.Eq(y, root) for root in y_roots])

# %%
y_ = y_roots[0]
display(sp.Eq(y, y_))

# %%
import numpy as np
import functools

@functools.lru_cache()
def Y_yr(s_, r_, yr_):
    def func(x_):
        #if (x_<0) or (x_>r_):
        #    return np.nan
        #else:
        if yr_ == s_: return lambda _: s_
        return sp.N(y_.subs({s:s_, r:r_, yr: yr_, x: x_}))
    return func

def S_yr(s_, r_, yr_, scipy = True):
    if yr_==s: return r_
    dS = sp.sqrt(1+y_.diff(x)**2).simplify()
    dS = dS.subs({s:s_, r:r_, yr: yr_}).simplify()
    if scipy:
        from scipy.integrate import quad
        dS_func = sp.lambdify(x, dS)
        return quad(dS_func, 0, r_)[0]
    return sp.N(sp.integrate(dS, (x, 0, r_)))

def V_yr(s_, r_, yr_, scipy = True):
    if yr_ > s_: raise ValueError("yr > s is not yet implemented")
    dV = y_.subs({s:s_, r:r_, yr: yr_})#dx
    if scipy:
        from scipy.integrate import quad
        dV_func = sp.lambdify(x, dV)
        return quad(dV_func, 0, r_, points = [0])[0]
    return sp.N(sp.integrate(dV, (x, 0, r_)))



# %%

@functools.lru_cache()
def yr_V(s_, r_, V_):
    V_min = V_yr(s_, r_, 0)
    V_max = s_*r_
    #if V_ == V_min: return V_min
    #if V_ == V_max: return V_max
    if V_<V_min: return np.nan
    if V_>V_max: return np.nan

    from scipy.optimize import brentq
    def fsolve(yr_):
        return V_yr(s_, r_, yr_)-V_
    yr_min = 0
    yr_max = s_
    root = brentq(fsolve, yr_min, yr_max)
    #if np.abs(Y_yr(s_, r_, root)(0)-s_)<1e-5: return np.nan
    return root


@functools.lru_cache()
def Y(s_, r_, V_):
    yr_ = yr_V(s_, r_, V_)
    print(f"yr = {yr_}")
    return Y_yr(s_, r_, yr_)

@functools.lru_cache()
def S(s_, r_,  V_):
    yr_ = yr_V(s_, r_, V_)
    y0 = Y(s_, r_, V_)(0)
    S0 = 0
    try:
        if y0<s_: S0 = s_ - y0
    except: return np.nan
    #print(f"yr = {yr_}")
    return S_yr(s_, r_, yr_)+S0