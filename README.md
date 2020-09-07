# Table Structure Recognition using Detectron2

## Introduction

This repository is based on detectron2 code: https://github.com/facebookresearch/detectron2/
Specifically, it contains additional code for training detectron2 for table structure recognition and trained models.

The models' performance is evaluated on different datasets by using different metrics: COCOmAP [[1]](#1), segments detection metrics [[2]](#2), [[3]](#3), adjacency relations based metrics(ICDAR-2013 [[4]](#4), ICDAR-2019 [[5]](#5)).


## Training and Validation Datasets

Training and validation datasets are built by using https://github.com/akmaral-yes/TableCellBank and
saved in the following way:

```
table_structure_recognition_with_detectron2
├── ...
└── datasets
    ├── ...
    └── real
        ├── train_tables
        ├── train_gt_tables_dict
        ├── val_tables
        └── val_gt_tables_dict
```

## Run the code

Build annotations for detectron2 based on ground truth from TableCellBank:
```shell
$ python detectron2_tables/generate_annotations0.py --dataclass column --mode val --multiproc
```

Visualize annotations:
```shell
$ python detectron2_tables/detectron2_tables.py --dataclass column --datatype real --visualize
```

Train a model:
```shell
$ python detectron2_tables/detectron2_tables.py --dataclass column --datatype real --train
```

Make predictions by trained models:
```shell
$ python detectron2_tables/detectron2_tables.py --dataclass column --datatype real --predict
$ python detectron2_tables/detectron2_tables.py --dataclass column --datatype ctdar --predict
```

Build ground truth for benchmark datasets (UNLV[[6]](#6), ICDAR[[4]](#4), CTDAR[[5]](#5)):
```shell
$ python metrics/unlv_build_gt.py
$ python metrics/icdar_build_gt.py
$ python metrics/ctdar_build_gt.py
```

Do post-processing (non-maximum suppression or rule-based):
```shell
$ python detectron2_tables/postproc_maskrcnn.py --dataclass column --datatype real --post nms
$ python detectron2_tables/postproc_maskrcnn.py --dataclass column --datatype ctdar --post rule
```

Compute COCOmAP or segments detection metrics:
```shell
$ python metrics/measures.py --measure map --dataclass column --datatype real 
$ python metrics/measures.py --measure segments --dataclass column --datatype real
```

Compute adjacency relations metric for ICDAR dataset[[4]](#4):
```shell
$ python metrics/icdar_adj_measure.py
```

Compute adjacency relations metric for CTDAR dataset[[5]](#5):
```shell
$ python metrics/ctdar_adj_measure.py
```

## References
<a id="1">[1]</a> 
[Lin et al., 2014] Lin, T.-Y., M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Doll´ar, and C. L.
Zitnick (2014). Microsoft coco: Common objects in context. In European conference on
computer vision, pp. 740–755. Springer.

<a id="2">[2]</a> 
[Shahab et al., 2010] Asif Shahab, Faisal Shafait, Thomas Kieninger, and Andreas Dengel.  An open approachtowards the benchmarking of table structure recognition systems.  InProceedings of the9th IAPR International Workshop on Document Analysis Systems, pages 113–120, 2010.

<a id="3">[3]</a> 
[Khan et al., 2020] Saqib Ali Khan, Syed Muhammad Daniyal Khalid, Muhammad Ali Shahzad, and FaisalShafait.  Table structure extraction with bi-directional gated recurrent unit networks.arXiv preprint arXiv:2001.02501, 2020.

<a id="4">[4]</a> 
[Goebel et al., 2013] Goebel, M., T. Hassan, E. Oro, and G. Orsi (2013). Icdar 2013 table competition. In 2013
12th International Conference on Document Analysis and Recognition, pp. 1449–1453. IEEE.

<a id="5">[5]</a> 
[Gao et al., 2019] Liangcai  Gao,  Yilun  Huang,  Herve  Dejean,  Jean-Luc  Meunier,  Qinqin  Yan,  Yu  Fang,Florian Kleber, and Eva Lang.  Icdar 2019 competition on table detection and recogni-tion (ctdar).  In2019 International Conference on Document Analysis and Recognition(ICDAR), pages 1510–1515. IEEE, 2019.

<a id="6">[6]</a> 
[Shahab, 2017] Shahab, A. "Table ground truth for the UW3 and UNLV datasets." (2017). 
http://www.iapr-tc11.org/mediawiki/index.php/Table_Ground_Truth_for_the_UW3_and_UNLV_datasets


