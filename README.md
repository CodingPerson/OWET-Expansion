# Open-World Entity Typing via LLM-Enhanced Hypercube Alignment

## Environment

* Computational platform: Pytorch 1.13.1, NVIDIA RTX A6000 GPU, CUDA Version 12.4
*  Development language: Python 3.8
* Libraries are listed in requirements.txt, which can be installed via the command `pip install -r requirements.txt`.

## Datasets

We construct two benchmark datasets of the OWET task based on existing fine-grained entity typing datasets (BBN, OntoNotes), which are provided in the folder `data`.

## LDHA for Type Recognition Subtask

#### Run LDHA on the BBN dataset:

```
sh init_BBN.sh 
```

This step generates a checkpoint file "init_file".
Based on this file, we run the following command to start training:

```
sh train_BBN.sh "init_file"
```

#### Run LDHA on the OntoNotes dataset:

```
sh init_OntoNotes.sh 
```

This step generates a checkpoint file "init_file".
Based on this file, we run the following command to start training:

```
sh train_OntoNotes.sh "init_file"
```

LDHA saves the checkpoint files “TP_path_BBN” and “TP_path_OntoNotes” for the BBN and OntoNotes datasets, respectively.

## HLTI for Ontology Enrichment Subtask

#### Run HLTI on the BBN dataset:
```
python top-down-search.py --TR_parh "TP_path_BBN" --dataset BBN
```
#### Run HLTI on the OntoNotes dataset:
```
python top-down-search.py --TR_parh "TP_path_OntoNotes" --dataset OntoNotes
```
