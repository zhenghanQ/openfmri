
# coding: utf-8

# In[22]:

#!/om/user/zqi/projects/CASL/Results/Imaging/openfmri/

import re
import os
import sys
from glob import glob

rootdir='/om/user/zqi/projects/CASL/Results/Imaging/openfmri/'
#get list of folders in rootdir
#list_items=os.listdir(rootdir)
list_items= [dI for dI in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir,dI))]


# In[23]:

len(list_items)
print(list_items)


# In[24]:

#regex for CASL131xx
regexCASL131=re.compile("^CASL131(?P<index>\d\d)")
regexCASL132=re.compile("^CASL132(?P<index>\d\d)")


# In[25]:

for item in list_items:
    if re.match(regexCASL131,item):
        m=re.search(regexCASL131,item)
        index=m.group('index')
        #process 131xx
        pre_items=os.listdir(rootdir+item)
        if not os.path.exists(rootdir+item+'/ses-pre'):
            os.mkdir(rootdir+item+'/ses-pre')
            for pi in pre_items:
                os.rename(rootdir+item+'/'+pi, rootdir+item+'/ses-pre/'+pi)
        pre_items_new=os.listdir(rootdir+item+'/ses-pre')
        pre_name=(item+'_ses-pre')
        for folder in pre_items_new:
            file_list=os.listdir(rootdir+item+'/ses-pre/'+folder)
            if not any(pre_name in s for s in file_list):
                for fi in file_list:
                    fi_new=fi.replace(item,pre_name)
                    os.rename(rootdir+item+'/ses-pre/'+folder+'/'+fi,rootdir+item+'/ses-pre/'+folder+'/'+fi_new)
        #process 132xx when applicable
        post_dir=rootdir+"CASL132{0}".format(index)
        id_post=("CASL132{0}".format(index))
        if os.path.exists(post_dir):
            post_items=os.listdir(post_dir)
            if not os.path.exists(rootdir+item+'/ses-post'):
                os.mkdir(rootdir+item+'/ses-post')
                for pi in post_items:
                    os.rename(post_dir+'/'+pi, rootdir+item+'/ses-post/'+pi)
            post_items_new=os.listdir(rootdir+item+'/ses-post')
            post_name=(item+'_ses-post')
            for folder in post_items_new:
                file_list=os.listdir(rootdir+item+'/ses-post/'+folder)
                if not any(post_name in s for s in file_list):
                    for fi in file_list:
                        fi_new=fi.replace(id_post,post_name)
                        os.rename(rootdir+item+'/ses-post/'+folder+'/'+fi,rootdir+item+'/ses-post/'+folder+'/'+fi_new)


# In[ ]:



