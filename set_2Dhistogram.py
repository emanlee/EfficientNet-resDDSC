import pandas as pd
from numpy import *
import json, re,os, sys
import h5py
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description="This script processes RNA-seq expression data to generate 2D histograms.")

parser.add_argument('-sc_gene_list', type=str, help='File containing gene symbol ID list of sc RNA-seq. If none, use "None".')
parser.add_argument('-gene_pair_label_file', type=str, help='File of the training gene pairs and their labels.')
parser.add_argument('-separation_index_file', type=str, help='File that indicates the separation index in the gene pair label file.')
parser.add_argument('-rpkm_data', type=str, help='HDF5 file containing scRNA-seq expression data. If none, use "None".')
#parser.add_argument('is_labelled', type=str, choices=['0', '1'], help='Flag indicating whether the gene pairs are labelled (1) or not (0).')

args = parser.parse_args()


save_dir = os.path.join(os.getcwd(),'2Dhistogram_data')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

def get_gene_list(file_name):
    import re
    h={}
    s = open(file_name,'r') #gene symbol ID list of sc RNA-seq
    for line in s:
        search_result = re.search(r'^([^\s]+)\s+([^\s]+)',line)
        h[search_result.group(1).lower()]=search_result.group(2) # h [gene symbol] = gene ID
    s.close()
    return h

def get_sepration_index (file_name):
    import numpy as np
    index_list = []
    s = open(file_name, 'r')
    for line in s:
        index_list.append(int(line))
    return (np.array(index_list))

# Script starts from here

if args.sc_gene_list != 'None':
    h_gene_list = get_gene_list(args.sc_gene_list)
    print('Read sc gene list')
else:
    print('No sc gene list')


if args.rpkm_data != 'None':
    store = pd.HDFStore(args.rpkm_data)   # scRNA-seq expression data
    rpkm = store['rpkm']
    store.close()
    print('Read sc RNA-seq expression')
else:
    print('No sc gene expression')

#Code used to read the breast cancer dataset.h5 file
# if not args.rpkm_data == 'None':
#     hdf5_path = args.rpkm_data  # Get the file path from the command line argument
#     # Initialize an empty list to hold the chunks
#     dataframe_chunks = []
# 
#     # Open the HDF5 file in read mode
#     with pd.HDFStore(hdf5_path, 'r') as store:
#         # Retrieve all the keys in the HDF5 file
#         hdf5_keys = [key for key in store.keys() if key.startswith('/rpkm_chunk_')]
#         # Sort the keys to maintain the original order
#         hdf5_keys.sort()
# 
#         # Loop over each key and read the corresponding DataFrame chunk
#         for key in hdf5_keys:
#             df_chunk = store[key]
#             dataframe_chunks.append(df_chunk)
# 
#     # Concatenate all chunks into a single DataFrame
#     rpkm = pd.concat(dataframe_chunks, axis=1)
#     print('Read sc RNA-seq expression from chunks')


########## generate 2Dhistogram
gene_pair_label = []
s=open(args.gene_pair_label_file)### read the gene pair and label file
for line in s:
    gene_pair_label.append(line)
gene_pair_index = get_sepration_index(args.separation_index_file)
s.close()
gene_pair_label_array = array(gene_pair_label)
for i in range(len(gene_pair_index)-1):   #### many sperations
    print (i)
    start_index = gene_pair_index[i]
    end_index = gene_pair_index[i+1]
    x = []
    y = []
    z = []
    for gene_pair in gene_pair_label_array[start_index:end_index]: ## each speration
        separation = gene_pair.split()
        x_gene_name,y_gene_name,label = separation[0],separation[1],separation[2]
        y.append(label)
        z.append(x_gene_name+'\t'+y_gene_name)

        if not args.sc_gene_list == 'None':
            x_tf = log10(rpkm[int(h_gene_list[x_gene_name])][0:43261] + 10 ** -2) # ## 43261 means the number of samples in the sc data, we also have one row that is sum of all cells, so the real size is 43262, that is why we use [0:43261]. For TF target prediction or other data, just remove "[0:43261]"
            x_gene = log10(rpkm[int(h_gene_list[y_gene_name])][0:43261] + 10 ** -2)# For TF target prediction, remove "[0:43261]"
            H_T = histogram2d(x_tf, x_gene, bins=32)
            H = H_T[0].T
            HT = (log10(H / 43261 + 10 ** -4) + 4) / 4
            x.append(HT)


    if (len(x)>0):
        xx = array(x)[:, :, :, newaxis]
    else:
        xx = array(x)
    save(save_dir+'/Nxdata_tf' + str(i) + '.npy', xx)
#    if args.is_labelled == '1':
    save(save_dir+'/ydata_tf' + str(i) + '.npy', array(y))
    save(save_dir+'/zdata_tf' + str(i) + '.npy', array(z))



