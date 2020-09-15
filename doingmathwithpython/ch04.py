# %%
'''
factor 是分解
Symbol 定义变量
solve 是解方程
'''
from sympy import Symbol, expand, factor, solve
from sympy.plotting import plot
# %%
x = Symbol('x')
# %%
(x+3)*(x-3)
# %%
# 分解
factors = factor(x**2 - 9)
# %%
# 展开
expand(factors)
# %%
factors
# %%
# 解方程
# %%
x = Symbol('x')
y = Symbol('y')
solve(x+38-9*y,dict=True)
# %%
plot(x**3-2,(x,-100,200))
# %%
