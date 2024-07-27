# EfficientNet-resDDSC

readme will come after the paper is published.


## Setting runtime environments:
Code is tested using Python >=3.7

It's better to create a virtual environment to run code
- conda create -n ddsc python=3.7 

activate the virtual environment 
- conda activate ddsc 

# install pacakages
- pip install pandas==1.3.5   
- pip install h5py==3.8.0  
- pip install tables==3.7.0  
- pip install numpy==1.21.5
- pip install tensorflow==2.11.0
- pip install keras==2.11.0
- pip install matplotlib==3.5.3
- pip install scipy==1.7.3
This experimental code runs on CPU.

## Data

We tested EfficientNet-resDDSC on the following three datasets.
- bone marrow-derived macrophages  
- dendritic cells 
- KEGG

scRNA-seq : 
    https://s3.amazonaws.com/mousescexpression/rank_total_gene_rpkm.h5
bone marrow drived macrophage scRNA-seq : 
    https://mousescexpression.s3.amazonaws.com/bone_marrow_cell.h5
dendritic single cell RNA-seq:
    https://mousescexpression.s3.amazonaws.com/dendritic_cell.h5
(Benchmark and processed gene expression profiles for bone marrow-derived macrophages, dendritic cells, KEGG are availabel from https://github.com/xiaoyeye/CNNC. )

We format the pairs with positive labels in the benchmark downloaded from the corresponding links, and randomly select same number of pairs with negative labels, to generate the training pair file (see folder data_evaluation).

To study the key gene BRCA1 gene in breast cancer patients, we downloaded the gene expression profiles of two untreated breast cancer patients (GSE123926) from NCBI database, and then constructed the relevant gene pairs of the BRCA1 gene and performed gene causality analysis using EfficientNet-resDDSC.


## TASK 1, Evaluating EfficientNet-resDDSC on three datasets

### STEP 1: Constructing two-dimensional histograms

**Code**: set_2Dhistogram.py

**Input**:
-sc_gene_list: containing gene symbol ID list of sc RNA-seq. If none, use "None".
-gene_pair_label_file: the training gene pairs and their labels.
-separation_index_file:  indicates the separation index in the gene pair label file.
-rpkm_data: HDF5 file containing scRNA-seq expression data. If none, use "None".
**Command for each cell type**:
```
python set_2Dhistogram.py -sc_gene_list ./data/sc_gene_list.txt -gene_pair_label_file ./data/ kegg_gene_pairs.txt  -separation_index_file ./data/ kegg_gene_pairs_num.txt -rpkm_data ./ rank_total_gene_rpkm.h5
```

**output**:

- x files: The representation of genes' expression files, used as the input of our model.
- y files: The labels for the corresponding pairs.
- z files: Indicate the gene names for each pair.


### STEP 2: Three-fold Cross-validation for EfficientNet-resDDSC

**Code**: EfficientNet-resDDSC.py

**Input**: the output of the STEP 1.

**usage**:

```
-length_TF: Number of data parts divided
-dataset_path: Path to the 2Dhistogram_data
-num_classes: Number of label classes

python EfficientNet-resDDSC.py -length_TF 9 -dataset_path ./input/den_2Dhistogram_data -num_classes 3 -output_path ./output/result.txt 

```

## TASK 2, Infer causality in breast cancer patients use EfficientNet-resDDSC

### STEP 1: Convert the downloaded dataset into h5 file. 

**Code**: makeh5.py

**Input**: 
The input dataset was a gene expression profile of two breast cancer patients downloaded from the NCBI database (GSE123926). Each patient dataset contains three files, barcodes.tsv, genes.tsv, and matrix.mtx.

**output**: 

The output is an .h5 file.

**usage**:

```
-barcodes: Path to the barcodes file
-genes: Path to the genes file
-matrix: Path to the gzipped matrix file
-output: Output path for the HDF5 file

python makeh5.py -barcodes ./data/PDX322/barcodes.tsv -genes ./data/PDX322/entrez_id.tsv -matrix ./data/PDX322/matrix.mtx.gz -output "D:/scRNA_seq/data1/PDX322/PDX322(1)_data.h5" 

```

### STEP 2: Constructing two-dimensional histograms

**Code**: set_2Dhistogram.py

**output**: 

The output is a breast_2Dhistogram_data file.

**usage**:

-sc_gene_list: File containing gene symbol ID list of sc RNA-seq. If none, use "None".
-gene_pair_label_file: File of the training gene pairs and their labels.
-separation_index_file: File that indicates the separation index in the gene pair label file.
-rpkm_data: HDF5 file containing scRNA-seq expression data. If none, use "None".

```
python set_2Dhistogram.py -sc_gene_list ./data/PDX322_SC_list.txt -gene_pair_label_file ./data/ brca1_pairs.txt -separation_index_file ./data/ brca1_pairs_num.txt -rpkm_data ./PDX322_data.h5
```

### STEP 3: Predict use trained EfficientNet-resDDSC

**Code**: predict.py

**usage**:

-length_TF: Number of data parts divided
-data_path: Path to the 2Dhistogram_data
-num_classes: Number of label classes
-model_path: Path to the trained model file
```
python predict.py -length_TF 1 -data_path ./data/breast_2Dhistogram_data -num_classes 3 -model_path ./trained_model/den_trained_model.h5
```
