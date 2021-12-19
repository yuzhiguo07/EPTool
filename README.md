# EPTool: A New Enhancing PSSM Tool for Protein Secondary Structure Prediction

Recently, a deep learning based enhancing PSSM method (Bagging MSA Learning (Y Guo, et al.)) has been proposed, and its effectiveness has been empirically proved. Program EPTool is the implementation of Bagging MSA Learning, which provides a complete training and evaluation workflow for the enhancing PSSM model. It is capable for handling different input dataset and various computing algorithms to train the enhancing model, then eventually improve the PSSM quality for those proteins with insufficient homologous sequences. In addition, EPTool equips several convenient applications, such as PSSM features calculator, and PSSM features visualization. In this paper, we propose designed EPTool, and briefly introduce its functionalities and applications. The detail accessible instructions is also provided. 

# EPTool Userguide

## preparation
Download Uniref50 fasta format database from https://www.uniprot.org/downloads to `./db/uniref50.fasta`

Donwload HMMER3.2.1 from http://hmmer.org/download.html and install the HMMER following the Userguide.pdf.

## Requirements:
cuda 10.2

python 3.6

pytorch 1.4.0

smile `pip install smile`

## Function
### Run jackhmmer
Run Jackhmmer following the Jackhmmer guide book(http://eddylab.org/software/hmmer/Userguide.pdf), and fit the output to the `aln_example/sample.aln` format. Here are the hmmer parameters:
```
phmmer -E 1 --domE 1 --incE 0.01 --incdomE 0.03 --mx BLOSUM62 --pextend 0.4 --popen 0.02 -o {out_path} -A {sto_path} --notextw --cpu {cpu_num} {fasta_path} {db_path}
```

### Calculate PSSM
```
python calculate_pssm.py --aln_path ./aln_example/sample.aln --save_path ./feat_example/sample.pssm --method 1 
```
#### Parameters
*aln_path* - MSA file path

*save_path* - save pssm feature file path

*method* - PSSM calculation method num, `0`, `1` or `2`, Usually using `1`.

*ss_path*(optional) - secondary structure label file path, see `./ss_example/sample.ss` for an example.

### Unsupervised model training example
```
python train.py --aln_dpath='./aln_example' --train_fname='sample.aln' --valid_fname='sample.aln' --model_path=./try01 2>&1 | tee try01.log
```
See comments for all hyper-parameters in `train.py`

This part of the manual will be completed upon acceptance of the EPTool paper.

See `./aln_example/sample.aln` for an example of input MSA file.

### Generate enhanced feature
```
python generate_new_feat.py --eval_feat_path='./feat_example/sample2.feat' --save_fpath='./feat_example/new.feat' --model_path='./try01' --epoch=1
```
See comments for all hyper-parameters in `generate_new_feat.py`

This part of the manual will be completed upon acceptance of the EPTool paper.

See `./feat_example/sample2.feat` for an example of input feat file.

### Generate PSSM Grayscale
```
python grayscale.py --pssm_path ./feat_example/sample.pssm
```
### Citation

Please cite the following paper in your publication if it helps your research:

Guo, Y., Wu, J., Ma, H., Wang, S. and Huang, J., 2020, May. Bagging MSA Learning: Enhancing Low-Quality PSSM with Deep Learning for Accurate Protein Structure Property Prediction. In International Conference on Research in Computational Molecular Biology (pp. 88-103). Springer, Cham.

Guo, Y., Wu, J., Ma, H., Wang, S., & Huang, J. (2021). EPTool: A New Enhancing PSSM Tool for Protein Secondary Structure Prediction. Journal of Computational Biology, 28(4), 362-364.


