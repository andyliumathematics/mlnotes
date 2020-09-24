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
import numpy as np

data=['a','b','c','a','a','b']
data1=np.array(data)
#计算信息熵的方法
def calc_ent(x):
    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
        print (ent)
calc_ent(data1) 

# %%
import numpy as np
data = np.array(['p','q','f','a','f','a','q','a','a','f','a'])
# %%
sdata = set(data)
sdata
# %%
logmid = 0.0
for i in sdata:
    logmid-=np.log(len(data[data==i])/len(data))
print(logmid/len(data))
# %%
import pandas as pd
import numpy as np
company=["A","B","C","X"]
data=pd.DataFrame({
    "company":[company[x] for x in np.random.randint(0,len(company),10)],
    "salary":np.random.randint(5,50,10),
    "age":np.random.randint(15,50,10)
}
)
# %%
data
# %%
group = data.groupby('company')
# %%
group
# %%
i = list(group)
# %%
i
# %%
i[0][1]
# %%
data.agg("sum")
# %%
r = group.agg({'salary':'median','age':'mean'})
# %%
type(r)
# %%
r
# %%
r['salary'].loc['A']
# %%
r.loc['A']['salary']
# %%
r.sort_values(by='company')
# %%
r
# %%
r.style
# %%
data
# %%
g=data.groupby('company', as_index=False)['salary'].count()
# %%
g['company'].count()