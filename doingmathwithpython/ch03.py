# %%
l = [38,32,49,15,806,806]
sum(l)
# %%
len(l)
# %%
sum(l)//len(l)
# %%
l.sort()
# %%
l
# %%

# %%
l.most_common(1)
# %%
'''
    求众数：出现频率最高的数据
'''
from collections import Counter
l = ['38','32','49','15','806','806']
c = Counter(l)
print(c.most_common()[0][0])
print(c.most_common(1))
print(c.most_common(2))
# %%
c.most_common()[0]
# %%
print(33)
# %%
