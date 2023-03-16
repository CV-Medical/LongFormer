# Longformer: Longitudinal Transformer for Alzheimer’s Disease Classification with Structural MRIs


Official implementation of [Longformer: Longitudinal Transformer for Alzheimer’s Disease Classification
with Structural MRIs](https://arxiv.org/pdf/2302.00901v2.pdf).


### Installation

install PyTorch 1.10.0 and torchvision 0.11.0:

```bash
pip install pytorch==1.10.0 torchvision==0.11.0
```

Install dependencies and pycocotools for VIS:

```bash
pip install -r requirements.txt
```

Compiling 3D-MSDA CUDA operators:

```bash
cd ./models/ops
sh ./make.sh
python test.py
```



### Data Preparation

Download and extract ADNI train and val images from [ADNI](https://adni.loni.usc.edu/) and annotations from [[baiduyun, code: 9cfu]](https://pan.baidu.com/s/1jobQZpR9zLBKGH2PZJ-_dg). We expect the directory structure to be the following:

```
/path/to/dataset/
├── images
│   ├── 002_S_0295
│   │   ├── 2006-04-18
│   │   │   ├── t1.nii.gz
│   │   ├── 2006-11-02
│   │   │   ├── t1.nii.gz
│   │   ├── 2007-05-25
│   │   │   ├── t1.nii.gz
│   │   ...
│   ├── 002_S_0619
│   │   ├── 2006-06-01
│   │   │   ├── t1.nii.gz
│   │   ├── 2006-12-13
│   │   │   ├── t1.nii.gz
│   │   ├── 2007-06-22
│   │   │   ├── t1.nii.gz
│   │   ...
│   ...
├── annotations
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
```
In annotation file, we annotate subject-wisely and construct the file format to be the following:

```
<subject name> <class> <num visits>
011_S_0005 0 5
022_S_0014 0 5
011_S_0016 0 5
067_S_0019 0 4
011_S_0021 0 11
...
```



##  



### Model zoo


#### ADNI model

Train on ADNI train set, evaluate on ADNI val set.       

| Model                                                        | Classification Type   |
| ------------------------------------------------------------ | ---- |
| Longformer_S [[baiduyun, code: yewe]](https://pan.baidu.com/s/1pYHtRpsGiT4gKPW6fqPprg) | NC v.s. AD |
| Longformer_L [[baiduyun, code: opb7]](https://pan.baidu.com/s/1nKtlCs_Oaa6qXOHRlExxSg) | NC v.s. AD |
| Longformer_S [[baiduyun, code: s6dc]](https://pan.baidu.com/s/1XtJex78TIGwbO77w-bMFfg) | sMCI v.s. pMCI |
| Longformer_L [[baiduyun, code: 2kno]](https://pan.baidu.com/s/1E_qbFyE2q4_7FBbnQFaNBA) | sMCI v.s. pMCI |


### Training

To train Longformer on ADNI, run:

```
python main.py 
--dataset_file ADNI \
--epochs 100 \ 
--lr 5e-4 \ 
--lr_drop 2 20 \ 
--num_workers 8 \ 
--backbone unet3d \ 
--output_dir output \ 
--batch_size 1 \  
--num_visits 3 \ 
--num_queries 300 \
--classification_type NC/AD \
--data_path /data/qiuhui/code/graph/data/ADNI/
```

The distributed version will be comming soon.


### Evaluation



Evaluating on ADNI val set:

(we provide a subset of preprocessed ADNI Val Set for evaluation, [[baiduyun, code: nlv0]](https://pan.baidu.com/s/1clHpfvRkPqYl3E5vZw7-ag))

```
python main.py 
--dataset_file ADNI \ 
--epochs 100 \ 
--lr 5e-4 \ 
--lr_drop 2 20 \ 
--num_workers 8 \ 
--backbone unet3d \ 
--output_dir output \ 
--batch_size 1 \  
--num_visits 3 \ 
--num_queries 300 \
--classification_type NC/AD \
--data_path /data/qiuhui/code/graph/data/ADNI/ \
--model ./output/checkpoint.pth \ 
--eval
```





## Acknowledgement

This repo is based on [Seqformer](https://github.com/wjf5203/SeqFormer) and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). Thanks for their wonderful works.
