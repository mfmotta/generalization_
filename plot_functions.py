# Ex: python plot_functions.py --add_to_file_name='1024-128' --path='/home/mariele/PycharmProjects/generalization/'
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('add_to_file_name', '', 'String to be added to file name')
flags.DEFINE_string('path', '/home/mariele/PycharmProjects/generalization/', 'path to project')


def plot(dicta,dictb,plot_title,legends):

    fig=plt.figure()
    fig.suptitle(plot_title, size = 18)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 16
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size

    i=1
    cols=3
    rows=int((len(dicta))/cols)
    for key in dicta:
        if key != 'sizes' and i<=rows*cols:
            pl=fig.add_subplot(rows,cols,i)
            pl.plot(dicta['sizes'],dicta[key],'.-',label=key[:-1]+'_'+legends[0])
            pl.plot(dicta['sizes'],dictb[key],'.-',label=key[:-1]+'_'+legends[1])
            pl.legend()
            i+=1



    fig.tight_layout(rect=[0, 0.04, 1, 0.95]) #left,bottom,right,topf
    title=plot_title.lower().replace('- ','').replace(' ','_').replace('=','')
    plt.savefig(os.path.join(FLAGS.path+'images/',title))
    plt.close()
    return


dictc1, dictc, dictr1, dictr= {}, {}, {}, {}
dicts = [dictr1, dictr, dictc1, dictc] #in this order

i=0
for name in ['random','correct']:
    for target in ['acc1','loss0']:
        with np.load(FLAGS.path + 'files/'+target+'_arrays_' + name + '.npz') as data:
            for item in data:  # [sizes, a_1, va_1, l_1, vl_1, l2_1, epcs_1,u_1]:
                if target == 'acc1' and item!='sizes':
                    renamed_item = item[:-1]  # names of arrays to be the same in two distinct dicts
                    dicts[i][renamed_item] = data[item]
                else:
                    renamed_item = item
                    dicts[i][item] = data[item]
        i+=1

plot(dicts[0],dicts[1],'Random Labels - Acc=1 vs Loss=0',['acc1','loss0'])
plot(dicts[0],dicts[1],'Random Labels - Acc=1 vs Loss=0',['acc1','loss0'])
plot(dicts[2],dicts[3],'Correct Labels - Acc=1 vs Loss=0',['acc1','loss0'])
plot(dicts[0],dicts[2],'Acc=1 - Random vs Correct Labels',['random','correct'])
plot(dicts[1],dicts[3],'Loss=0 - Random vs Correct Labels',['random','correct'])
