
##最终创建.h5的代码
import pandas as pd
import h5py
from scipy.io import mmread
import gzip
import numpy as np
import os

# File paths for the uploaded
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description="This script processes scRNA-seq expression data and stores it in HDF5 format.")
parser.add_argument('-barcodes', type=str, required=True, help='Path to the barcodes file.')
parser.add_argument('-genes', type=str, required=True, help='Path to the genes file.')
parser.add_argument('-matrix', type=str, required=True, help='Path to the gzipped matrix file.')
parser.add_argument('-output', type=str, required=True, help='Output path for the HDF5 file.')
args = parser.parse_args()

barcodes_path = args.barcodes
genes_path = args.genes
matrix_path = args.matrix


# Read the barcodes and genes files
barcodes_df = pd.read_csv(barcodes_path, header=None, sep='\t')
barcodes = barcodes_df[0].astype(str).tolist()

genes_df = pd.read_csv(genes_path, header=None, sep='\t')
genes = genes_df[0].astype(str).tolist()

# Read the matrix file
with gzip.open(matrix_path, 'rb') as matrix_file:
    matrix = mmread(matrix_file).tocsc()

# Check if the matrix dimensions match the lengths of the genes and barcodes lists
if matrix.shape[0] != len(genes) or matrix.shape[1] != len(barcodes):
    raise ValueError("The dimensions of the matrix do not match the number of genes and barcodes.")

# Transpose the matrix if necessary to match genes (rows) x barcodes (columns)
if matrix.shape[0] == len(barcodes) and matrix.shape[1] == len(genes):
    matrix = matrix.transpose()

# Create a DataFrame with genes as rows and cells as columns
df = pd.DataFrame(matrix.toarray(), index=genes, columns=barcodes)
df = df.transpose()  # Now, df has cells as rows and genes as columns
# Define the HDF5 file path
h5_file_path = args.output


# Remove the existing HDF5 file if it exists to start fresh
if os.path.exists(h5_file_path):
    os.remove(h5_file_path)
num_chunks = 10
chunk_size = (df.shape[1] + num_chunks - 1) // num_chunks

# Save each chunk to the HDF5 file
for chunk in range(num_chunks):
    start_col = chunk * chunk_size
    end_col = min((chunk + 1) * chunk_size, df.shape[1])
    df_chunk = df.iloc[:, start_col:end_col]
    chunk_key = f'rpkm_chunk_{chunk}'
    df_chunk.to_hdf(h5_file_path, key=chunk_key, mode='a', format='table')

# Read and concatenate all chunks back into a single DataFrame (if needed)
df_full = pd.DataFrame()
for chunk in range(num_chunks):
    chunk_key = f'rpkm_chunk_{chunk}'
    df_chunk = pd.read_hdf(h5_file_path, chunk_key)
    df_full = pd.concat([df_full, df_chunk], axis=1)


