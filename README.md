## Multivariate Activation Functions for Sparse Regularizer Learning (MAF_SRL)

### Datasets Descriptions

- Original similarity matrices are stored in **/datasetW**, which are generated by KNN (see Matlab codes in ConstructW.zip).
- Original multi-view datasets are stored in **/_multiview datasets**.

### Environment

Require Python 3.8

- torch 1.8.0
- numpy 1.16.3
- tqdm 4.28.1
- scikit-learn 0.20.3

### Quick Running

- Run  `python ./run_Clustering.py --dataset-id 1` for clustering tasks.
- Run  `python ./run_Classification.py --dataset-id 1` for semi-supervised classification tasks.
- **Note:** dataset-id values of all presented datasets are as shown below:
  - `1:'ALOI', 2:'Caltech101-7', 3:'Caltech101-20', 4:'MNIST', 5:'MSRC-v1', 6:'NUS-WIDE', 7:'Youtube', 8:'ORL'`

