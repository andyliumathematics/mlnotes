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
fadf
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
# %%dfasd
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
# %%
company=["A","B","C"]
data=pd.DataFrame({
    "company":[company[x] for x in np.random.randint(0,len(company),10)],
    "salary":np.random.randint(5,50,10),
    "age":np.random.randint(15,50,10)
}
)
# %%
data['avg_salary'] = data.groupby('company')['salary'].transform('sum')
# %%
data
# %%
data.groupby('company')['salary'].mean().to_dict()
# %%
data['company'].map(data.groupby('company')['salary'].mean().to_dict())
# %%
data[["height","weight","age"]].apply(np.sum, axis=0)
# %%
[3,3,4]+[3,5,6]
# %%
b = ['saw', 'small', 'He', 'foxes', 'six']
# %%
b.sort(key=len)# %%

# %%
b

# %%
b.sort()
# %%
b
# %%
import bisect
bisect.bisect(b,2)
# %%
list(zip([3,3,4],[3,5,6]))
# %%
hash([3,3,4])
# %%
a = {3,3,4}
b={3,5,6}
# %%
a|b
# %%
{3,3,4}|{3,5,6}
# %%
import math
math.log(3.17828,2)
# %%
math.log2(3.17828)
# %%
math.log1p(1)
# %%
math.log(2)
# %%
l=[]
l.append([32])
l[0]
# %%
l=[[1,2,3],[1,2,3]]
# %%
set(l)
# %%
import math
0.5*math.log2(0.5)*2
# %%

import math
0.3*math.log2(0.3)*3 +0.1*math.log2(0.1)
# %%
1*math.log2(1)
# %%

0.5*math.log(0.5)*2
# %%
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(53)
# %%
np.ran
# %%
import numpy as np
# %%
data = {i:np.random.randn() for i in range(7)}
# %%
data
# %%
import pandas as pd
m = pd.Series({1:'2',3:4,5:6,"3":3})


# %%
m["3"]
# %%
m.dtype
# %%
pd.NaT
# %%
import numpy as np
import matplotlib.pyplot as plt
rg = np.arange(10)
# %%
rg
# %%
import seaborn as sn
plt.plot(rg)
# %%
3+(4*33j)**2
0x32# %%

# %%
import numpy as np
np.sort ([1,8,3]+[4,5,6]).size()
type([])
# %%
def findMedianSortedArrays( nums1, nums2):
    numsnew = np.sort(nums1+nums2)
    if(numsnew.size%2==1):
        return numsnew[numsnew.size/2]+0.0
    else:
        return (numsnew[numsnew.size/2-1]+numsnew[numsnew.size/2])/2.0
# %%
type([])
# %%
x = np.ndarray([3,4,5])
# %%
x[1]
# %%
mm = np.ndarray([23,34,6,234]+[3,4,5])
# %%
mm = np.array([32,3,23,5,34])
# %%
mm
# %%
mm = np.sort([23,4,5,4]+[324,5,3,354,4])
# %%
mm.size/2
# %%
mm[int(mm.size/2)]
# %%
