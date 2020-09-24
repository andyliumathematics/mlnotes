# %%
import pandas as pd
import numpy as np
# %%
boolean=[True,False]
gender=["男","女"]
color=["white","black","yellow"]
data=pd.DataFrame({
    "height":np.random.randint(150,190,100),
    "weight":np.random.randint(40,90,100),
    "smoker":[boolean[x] for x in np.random.randint(0,2,100)],
    "gender":[gender[x] for x in np.random.randint(0,2,100)],
    "age":np.random.randint(15,90,100),
    "color":[color[x] for x in np.random.randint(0,len(color),100) ]
}
)
# %%
data
# %%
data.head()
# %%
data['gender']=data["gender"].map({'男':1,'女':0})
# %%
data['gender']=data["gender"].map({1:'男',0:'女'})
# %%
data
# %%
data[['age','weight']].apply(np.sum,axis=1)
# %%
data[['age','weight']].applymap(lambda x :'np.sum(%.2lf,32)' % x)
# %%
data.merge(data,how='inner',on=['gender','age'])
# %%
data
# %%
