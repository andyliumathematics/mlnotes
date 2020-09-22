# %%
print("adf")
# %%
t = 2,3,4,'23'
# %%
t
# %%
type(t)
# %%
t?
# %%
iter(t)
# %%
None
# %%
a = None
# %%
a

# %%
a =89
# %%
import numpy as np
a is 89
# %%
a = [1,2,3,4,5]
# %%
a

# %%
x,y,z,t,m = a
# %%
x
# %%
values = 2,34,3,412,3
# %%
values
# %%
a,b,*_ = values
# %%
x = range(10)
# %%
list(x)
# %%
tuple(x)
# %%
import numpy as np 
np.arange(100)
# %%
type(np.arange(100))
# %%
a = np.empty((4,3,4,2,3))
# %%
a.dtype
# %%
type(a)
# %%
list(np.arange(32,300))
# %%
np.random.randn(4)
# %%
np.arange(32)
# %%
import pandas as pd
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
# %%
obj2
# %%
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],'key2' : ['one', 'two', 'one', 'two', 'one'],'data1' : np.random.randn(5), 'data2' : np.random.randn(5)})
# %%
df
# %%
grouped = df['data1'].groupby(df['key1'])
# %%
grouped.mean()
# %%
