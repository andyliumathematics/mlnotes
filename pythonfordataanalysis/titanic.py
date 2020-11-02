# %%
import pandas as  pd
import numpy as np

# %%
data_train =pd.read_csv(r"train.csv")
# %%
data_train.info
# %%
group = data_train.groupby('Sex')
# %%
group.agg('count')

# %%
data_train['Sex2'] = data_train['Sex'].map({'female':0,'male':1})
# %%
data_train
# %%
def f():
    return 1,2,3
# %%
x = f()
# %%
x = 3,4,5
x
# %%

# %%
f = lambda x:x**2
# %%
print(f(8))
# %%
# %%                                
strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
# %%
strings.sort(key = lambda x: x*3)
# %%
strings
# %%
# %%
strings, lambda x:  x+'2'
# %%
f= lambda x:x+'2'
# %%
f('3')
# %%
from datetime import datetime
# %%
datetime.now()
# %%
delta = datetime(2020,5,3) - datetime(2020,1,1,3)
# %%
delta.days
# %%
delta.seconds
# %%
delta

# %%
from datetime import timedelta
# %%
timedelta(23,seconds=3)
# %%
datetime.strptime('2020-09-09','%Y-%m-%d')
# %%
from datautil.python-parser import parse
# %%
parse('2020-09-20')
# %%
import pandas as  pd
# %%
pd.to_datetime('2020 08 1')
# %%
from vnpy import *
class Doc(object):
    def wang(self):
        self.aa = "adflakdj"
        print("alkdfladj")
# %%
d =  Doc()
# %%
d.wang()
# %%
d.aa
# %%
Doc().wang().aa
# %%

# %%
for i in enumerate([3,43,4,5,43,23,2]):
    print(i)
# %%
[3,4,5].sort()

# %%
l = [3,45,3,5,7,657,234,45,2,4,0x33]
# %%
l
l2 = ['32','234234','32','asdfadddd']
# %%
sorted(l2,key=len,reverse=True)
# %%
l
# %%
sq1 = [3,4,5]
sq2 = [34,5,6,32,32]
# %%
l = list(zip(sq1,sq2))
# %%
a ,b = zip(*l)
# %%
a
# %%
b
# %%
di = {1:2,2:53,33:3}
# %%
del di[1]
# %%
di
# %%
