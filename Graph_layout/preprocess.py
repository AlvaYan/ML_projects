
n_zip=2#number of zipped data minus one, or the largest suffix
n_inzip=5#500,number of samples in one zip 
path_data1="D:\\Career\\DS-ML\\1cademy\\research\\data\\dataset2\\"
path_data2="D:\\Career\\DS-ML\\1cademy\\research\\data\\preprocessed\\"

import dgl
from dgl.data import DGLDataset
import torch
import os
import pickle
import numpy as np
import itertools
import zipfile

def process_single_graph(data):#input is str directly read from one txt file
    #-----------data cleaning and rearranging--------
    data=data.replace('true','True')
    data=data.replace('false','False')
    data='g='+str(data)
    lcls = locals()
    exec(data, globals(), lcls )
    g = lcls["g"]
    #------------------------ process to attributes
    G = dgl.DGLGraph()
    G.add_nodes(g['_nodeCount'])
    in_=[]
    out_=[]
    for e in g['_edgeObjs'].keys():
        out_.append(int(g['_edgeObjs'][e]['v'][1:]))
        in_.append(int(g['_edgeObjs'][e]['w'][1:]))
    out_=torch.tensor(out_,dtype=torch.long)
    in_=torch.tensor(in_,dtype=torch.long)
    G.add_edges(out_,in_)
    #scale both length and position with output canvas width and height.
    #This way, the trained model assume that the canvas is bounded, but the input size fit just right with canvas size
    width=g['_label']['width']
    height=g['_label']['height']

    #scale both length and position with output canvas width and height.
    #This way, the trained model assume that the canvas is boundless
    width=1500
    height=1500

    x_=[(g['_nodes']['n'+str(n)]['width']/width,g['_nodes']['n'+str(n)]['height']/height) for n in range(g['_nodeCount'])]
    #for xx in x_:
    #  if int(xx[1])==0:
    #    print(("degenerate box",i,j))
    #    print(x_)
    #    break
    #x_=[(g['_nodes']['n'+str(n)]['width'],g['_nodes']['n'+str(n)]['height']) for n in range(g['_nodeCount'])]
    y_=[(g['_nodes']['n'+str(n)]['x']/width,g['_nodes']['n'+str(n)]['y']/height) for n in range(g['_nodeCount'])]
    #y_=[(g['_nodes']['n'+str(n)]['x'],g['_nodes']['n'+str(n)]['y']) for n in range(g['_nodeCount'])]

    y_=list(itertools.chain.from_iterable(y_))
    #y_=torch.tensor(y_,dtype=torch.double)
    #print(y_)
    G.ndata['x']=torch.tensor(x_,dtype=torch.double)
    G.edata['x']=torch.tensor(np.zeros((g['_edgeCount'],1)),dtype=torch.double)
    return (G,y_)

#--------iterate over all files to construct graph dataset--------
sfix=['']+[' ('+str(i+1)+')' for i in range(n_zip)]#filename suffix
graphs = []
labels = []
times=[]
ct=0
os.chdir(path_data1)
for i in sfix:
    zipname='training'+i+'.zip'
    archive = zipfile.ZipFile(zipname, 'r')
    for j in range(n_inzip):
        ct+=1
        #-----------load data----
        filename='graph'+str(j)+'.txt'
        data = (archive.read(filename).decode("utf-8"))
        filename='time'+str(j)+'.txt'
        time = (archive.read(filename).decode("utf-8"))
        time=(time.split(','))
        #----------process and add to dataset
        (g,label)=process_single_graph(data)
        graphs.append(g)
        labels.append(label)
        times.append(time)
# make sure all sublists have same dimension and convert it torch tensor
# Convert the label list to tensor for saving.
def trim(y):
    y_len=[len(_) for _ in y]
    y_dim=max(y_len)
    y_dim=256
    yy=([_+[0 for _ in range(y_dim-len(_))] for _ in y])
    return yy
labels=trim(labels)
labels=torch.tensor(labels,dtype=torch.double)

os.chdir(path_data2)
filehandler = open("graphs.obj","wb")
pickle.dump(graphs,filehandler)
filehandler.close()
filehandler = open("labels.obj","wb")
pickle.dump(labels,filehandler)
filehandler.close()
filehandler = open("times.obj","wb")
pickle.dump(times,filehandler)
filehandler.close()
print(times)

