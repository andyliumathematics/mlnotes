# %%
# 除法 取的小数结果
3/2
# %%
# 除法 整除 最后结果去掉小数部分
3//2
# %%
# 分数表示

from fractions import Fraction
f = Fraction(38/32)
g = Fraction(38,32)
h = Fraction(3832.323)
print(f)
print(g)
print(h)
# %%
# 复数
i= 3+4j
# %%
from sympy import Symbol, expand, factor, solve
from sympy.plotting import ploi
# %%
x = Symbol('x')
# %%
x
# %%
