# EPCR
A PyTorch implementation for the paper [**Consistency Regularization for Deep Face Anti-Spoofing**] by Zezheng Wang


### Dependencies

If you don't have python 3.8 environment:
```
conda create -n EPCR python=3.8
conda activate EPCR
```
Then install the required packages:
```
pip install -r requirements.txt
```

### Download OCIM datasets, preprocess and prepare the train/test list


### Run

#### Run this command to train DG with TransFAS architecture and with normal protocols
```
bash configs/run_ocim_epcr_ssdg_transfas_cim-o.sh
```

#### Run this command to train DG with semi-supervised protocols of leave-one-out protocols.
```
bash configs/run_ocim_leave_one_om-i_c.sh
```

#### Run this command to train DG with semi-supervised protocols of intra-labeled protocols.
```
bash configs/run_ocim_semi_Intra_live0.2_spoof0.2.sh
```

### Semi-Supervised protocols
https://pan.baidu.com/s/1prFW1NMRwtutpugaqlbr1Q   code: pfix

### Pretrained odels
https://pan.baidu.com/s/1qL9xUGOuAJbyP4ppXdy44A   code: jdfk