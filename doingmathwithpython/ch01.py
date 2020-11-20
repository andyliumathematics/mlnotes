# %%
3/2
# %%
# 除后去掉小数部分
3//2
# %%
# 求余数
7%3
# %%
# 幂
4**3

# %%
34**(1/3)
# %%
8**(1/3)
# %%
from fractions import Fraction

from numpy.lib.type_check import iscomplex
f = Fraction(38/32)
# %%
a = 2+3j

# %%
a
# %%
type(a)
# %%
a.real
# %%
a.imag
# %%
# 共轭复数
a.conjugate()
# %%
abs(a)
# %%
a='3'
# %%
int(a)+2
# %%
int(265538)
# %%
int(2.0)

# %%
a=float(3)
# %%
a
# %%
1.0.is_integer()
# %%
ff = Fraction(input("alkdjf"))
# %%
# range和np.linspace的区别是：
# range(a,b) 表示[ab)区间整数的序列
# np.linspace(a,b)表示[a,b]闭区间，中50个间隔相等的float的序列，50是可设置的，参数中可设置num=50
for i in range(1,32):
    print(i)

import numpy as np
for i in np.linspace(2,3):
    print(i)
# %%
from fractions import Fraction
f = Fraction(3,3)
# %%
f
# %%
4/3
# %%
type(4/2)

# %%
type(4)
# %%
3//4

# %%
-3//4
# %%
from fractions import Fraction
8**Fraction(30,4)
# %%
int(-0.34)
# %%
int(Fraction(3,23))
# %%
p = 3.8j
# %%
3.8J*4.2j
# %%
8J
# %%
p.real
# %%
p.imag
# %%
p.conjugate()
# %%
(8).conjugate()
# %%
abs(88+8j)
# %%
a = input()
# %%
type(a)
# %%
int(float('2.0'))
# %%
int(2.0)
# %%
(3).is_integer()
# %%
(1).is_integer()
# %%
Fraction(3,0)
# %%
from matplotlib import pyplot as plt
plt.plot([1,2,34,2,123,5],marker='o')

# %%
from sympy import Symbol
# %%
xx = Symbol('tx')
# %%
xx*8
# %%
yy = Symbol('ff')
# %%
xx*yy+83
# %%
x = Symbol('x')
# %%
y = Symbol('y')
# %%
 xy = x **2 -y **2
# %%
from sympy import factor
# %%
fa = factor(factor(xy))
# %%
from sympy import expand
expand(fa)
# %%
