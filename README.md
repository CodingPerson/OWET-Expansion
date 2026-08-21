# The Framework DVHA

## Environment

* Computational platform: Pytorch 1.13.1, NVIDIA RTX A6000 GPU, CUDA Version 12.4
*  Development language: Python 3.8
* Libraries are listed in requirements.txt, which can be installed via the command `pip install -r requirements.txt`.

## Datasets

We construct two benchmark datasets of the OWET task based on existing fine-grained entity typing datasets (BBN, OntoNotes), which are provided in the folder `data`. The dataset statistics are shown as follows:

|            **Dataset**             | **BBN** | OntoNotes |
| :--------------------------------: | :-----: | :-------: |
|          **Known types**           |   34    |    43     |
|         **Unknown types**          |   11    |    14     |
|       **Training instances**       |  4641   |   3505    |
|      **Validation instances**      |  1154   |    863    |
|  **Known-type testing instances**  |  5775   |   4348    |
| **Unknown-type testing instances** |  1712   |   1758    |

## Reproduce

#### Run DVHA on the BBN dataset:

```
sh init_BBN.sh 
```

This step generates a checkpoint file "init_file".
Based on this file, we run the following command to start training:

```
sh train_BBN.sh "init_file"
```

#### Run DVHA on the OntoNotes dataset:

```
sh init_OntoNotes.sh 
```

This step generates a checkpoint file "init_file".
Based on this file, we run the following command to start training:

```
sh train_OntoNotes.sh "init_file"
```
