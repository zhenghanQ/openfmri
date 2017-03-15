
# coding: utf-8

# In[1]:

#!/om/user/zqi/projects/CASL/Results/Imaging/openfmri/

import re
import os
import sys
from glob import glob

rootdir='/om/user/zqi/projects/CASL/Results/Imaging/openfmri/'
#get list of folders in rootdir
#list_items=os.listdir(rootdir)
list_items= [dI for dI in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir,dI))]


# In[2]:

len(list_items)
print(list_items)


# In[3]:

#regex for CASL131xx
regexCASL131=re.compile("^CASL131(?P<index>\d\d)")
regexCASL132=re.compile("^CASL132(?P<index>\d\d)")


# In[8]:

for item in list_items:
    if re.match(regexCASL131,item):
        os.rename(rootdir+item, rootdir+'sub-'+item)


# In[7]:

item='CASL13100'
re.match(regexCASL131,item)
print(rootdir+'sub-'+item)


# In[ ]:



