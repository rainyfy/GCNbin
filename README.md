# GCNbin

GCNbin is a multi-view and deep learning-based metagenomic contigs binning tool, GCNbin performs synchronous feature learning for compositional values and structural graphs by autoencoder and graph convolutional network respectively, and uses a dual self-supervised module to unify the results of the two neural networks and to guide the binning task. GCNbin is tested on contigs obtained from next-generation sequencing (NGS) data. Currently, GCNbin supports contigs assembled using metaSPAdes and MEGAHIT. You can contact us via liuyun313@jlu.edu.cn.

## Dependencies
- scipy - version 1.3.1
- numpy - version 1.17.2
- biopython - version 1.7.4
- torch -version 1.4.0
- scikit_learn 1.1.2
- h5py 2.10.0
- pandas 1.1.5

GCNbin uses the script ``prep_graph``in [METAMVGL](https://github.com/ZhangZhenmiao/METAMVGL) to build the PE graph, whcih dependencies:
- GCC with C++11, HTSlib.

GCNbin uses the following tools to scan for single-copy marker genes. These tools have been tested on the following versions.
- FragGeneScan - version 1.31
- HMMER - version 3.3.2

## Assembly Graph

For metaSPAdes, the assembly graph (assembly_graph.fastg) is already in the output folder.
For MEGAHIT, the assembly graph is derived from final.contigs.fa:
```
megahit_toolkit contig2fastg <k_mer> final.contigs.fa > final.contigs.fast
```

## Run GCNbin


```
usage: GCNbin [-h] --assembler ASSEMBLER --graph GRAPH --contigs CONTIGS --bam  BAM --output OUTPUT
[--abundance ABUNDANCE] [--paths PATHS] [--K K] [--prefix PREFIX] [--min_length MIN_LENGTH] [--topk TOPK] [--lambda LAMBDA] 
[--learn_rate LEARN_RATE] [--h1 H1] [--h2 H2] [--z Z] [--nthreads NTHREADS] [-v] 
optional arguments:  
-h, --help                                show this help message and exit   
--assembler ASSEMBLER                     name of the assembler used (Supports Spades, MEGAHIT)  
--graph GRAPH                             path to the assembly graph file   
--contigs CONTIGS                         path to the contigs file   
--bam BAM                                 path to the bam file   
--output OUTPUT                           path to the output folder   
--abundance ABUNDANCE                     path to the abundance file   
--paths PATHS                             path to the contigs.paths file (only in SPAdes)   
--K K                                     cluster number [default: auto]   
--prefix PREFIX                           prefix for the output file   
--min_length MIN_LENGTH                   minimum length of contigs to consider for binning [default: 300]   
--topk TOPK                               number of K in KNN graph. [default: 1]   
--lambda LAMBDA                           parameter controling the discards of low-confidence contigs [default: 0]   
--learn_rate LEARN_RATE                   learn rate of model [default: 1e-3]   
--h1 H1                                   parameter of neural network[default: 100]  
--h2 H2                                   parameter of neural network[default: 70]   
--z Z                                     parameter of neural network [default: 50]   
--nthreads NTHREADS                       number of threads to use [default: 8]   
-v, --version                             show program's version number and exit
```
Example 1: Run for metaSpades assembly:
```
GCNbin --assembler spades --graph ./contigs.fastg --contigs ./contigs.fasta --bam ./allsample.bam --output ./result/ --abundance ./coverage.tsv --paths ./contigs.paths --lambda 1 --nthreads 24
```
Example 2: Run for MEGAHIT assembly:
```
GCNbin --assembler megahit --graph ./final.contig.fastg --contigs ./final.contig.fa --bam ./allsample.bam --output ./result/ --abundance ./coverage.tsv --lambda 1 --nthreads 24
```

## reference
We modify or use certain code or scripts from [MetaCoAg](https://github.com/metagentools/MetaCoAG) and [METAMVGL](https://github.com/ZhangZhenmiao/METAMVGL).
