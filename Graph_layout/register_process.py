import numpy as np
import itertools

#-------------------raw sample graphs to input item------
#---------Python script that load self defined dgl dataset----------------
#------------------Define parameters and import packages---------------------------------
n_zip=12#12,35,number of zipped data minus one, or the largest suffix
n_inzip=500#500,number of samples in one zip 
path_data="/afs/crc.nd.edu/user/t/tyan/my_project_dir/data/dataset2/"

import dgl
from dgl.data import DGLDataset
import torch
import os

#---------------define dataset------------------
class ONECademyDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='ONECademy')

    def process(self):
        
        #-------------- define function that process a single graph to DGLGraph object-----
        def process_single_graph(data):#input is str directly read from one txt file
            #-----------data cleaning and rearranging--------
            data=data.replace('true','True')
            data=data.replace('false','False')
            data='g='+data
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
        sfix=['']+['_'+str(i+1) for i in range(n_zip)]#filename suffix
        self.graphs = []
        self.labels = []
        for i in range(n_inzip):
            for j in sfix:
                #-----------load data----
                filename='graph'+str(i)+j+'.txt'
                os.chdir(path_data)
                text_file = open(filename, "r")
                data = text_file.read()
                text_file.close()
                #----------process and add to dataset
                (g,label)=process_single_graph(data)
                self.graphs.append(g)
                self.labels.append(label)

        # make sure all sublists have same dimension and convert it torch tensor
        # Convert the label list to tensor for saving.
        def trim(y):
            y_len=[len(_) for _ in y]
            y_dim=max(y_len)
            y_dim=256
            yy=([_+[0 for _ in range(y_dim-len(_))] for _ in y])
            return yy
        self.labels=trim(self.labels)
        self.labels=torch.tensor(self.labels,dtype=torch.double)
    def __getitem__(self, i):
        #print(self.graphs[i].ndata['x'])
        #print(self.graphs[i].ndata['x'].size())
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


#-----------------------registration------------------------------

import os
#os.chdir('/afs/crc.nd.edu/user/t/tyan/my_project_dir/Graphormer/examples/customized_dataset/')
from graphormer.data import register_dataset
import numpy as np
from sklearn.model_selection import train_test_split

@register_dataset("ONECademy")
def create_customized_dataset():
    dataset = ONECademyDataset()
    #print("1C data loaded")
    num_graphs = len(dataset)
    #print(num_graphs)
    # customized dataset split
    train_valid_idx, test_idx = train_test_split(
        np.arange(num_graphs), test_size=num_graphs // 10, random_state=0
    )
    train_idx, valid_idx = train_test_split(
        train_valid_idx, test_size=num_graphs // 5, random_state=0
    )
    return {
        "dataset": dataset,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "source": "dgl"
    }

print("finihsed loading data")