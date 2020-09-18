# %%
import os
d = os.listdir("D:\workproject")
for i in d:
    rundir = "D:\workproject\\"+i
    print(rundir)
    os.system("cd "+ rundir)
    os.system(" git config --local user.name liulei ")
    os.system(" git config --local  user.email liulei.bj@fang.com ")
    os.system(" dir ")
# %%
