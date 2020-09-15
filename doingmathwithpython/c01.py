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
