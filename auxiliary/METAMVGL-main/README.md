# METAMVGL

## Dependencies
- gcc with C++11 support, [htslib](https://github.com/samtools/htslib) required
- python3 with [numpy](https://numpy.org/install/), [scipy](https://www.scipy.org/install.html) and [networkx](http://networkx.github.io/)

## Install
Before installing, make sure [htslib](https://github.com/samtools/htslib) is properly installed, with ```/path/to/htslib/include``` in ```$CPLUS_INCLUDE_PATH```, ```/path/to/htslib/lib``` in both ```$LIBRARY_PATH``` and ```$LD_LIBRARY_PATH```. To install METAMVGL:
```
git clone https://github.com/ZhangZhenmiao/METAMVGL.git
cd METAMVGL && make && chmod +x METAMVGL.py
```
The commands will generate ```prep_graph```. We need add it to the environmental variables:
```
export PATH=`pwd`:$PATH
```

## Usage

### Assembly

Currently we support [metaSPAdes](https://github.com/ablab/spades) and [MEGAHIT](https://github.com/voutcn/megahit).

### Initial binning

We accept any initial binning tools. To convert the binning result to the input format of METAMVGL, we suggest to use [prepResult.py](https://github.com/Vini2/GraphBin/tree/master/support):
```
python prepResult.py --binned /path/to/folder_with_binning_result --assembler assembler_type_(SPAdes/MEGAIHT) --output /path/to/output_folder
```
It will create a file ```initial_contig_bins.csv``` in ```/path/to/output_folder```.

### Prepare graphs

For MEGAHIT, the assembly graph in .fastg format is derived from final.contigs.fa:
```
megahit_toolkit contig2fastg k_mer final.contigs.fa > final.contigs.fastg
```

We generate the assembly graph (.ag) and PE graph (.pe) by ```prep_graph```:
```
usage: prep_graph --assembler=string --assembly-graph=string --bam=string --output=string [options] ...
options:
  -a, --assembler          the assembler used to produce contigs, currently support metaSPAdes and MEGAHIT (string)
  -c, --contigs            the path to the contigs, only needed for MEGAHIT (string [=final.contigs.fa])
  -p, --paths              the path to the .paths file, only needed for metaSPAdes (string [=contigs.paths])
  -g, --assembly-graph     the path to the assembly graph in fastg (string)
  -b, --bam                the path to the alignment bam file (string)
  -m, --mapping-quality    the threshold of mapping quality (double [=10])
  -i, --identity           the threshold of identity (double [=0.95])
  -s, --insert-size        the insert size of paired-end reads (int [=270])
  -n, --pe                 the minimum number of paired-end reads to support a link (int [=3])
  -o, --output             the prefix to output (string)
  -?, --help               print this message
```

### Multi-view graph-based binning
We create the binning result by ```METAMVGL.py```:
```
usage: METAMVGL.py [-h] --contigs CONTIGS --assembler ASSEMBLER
                   --assembly_graph ASSEMBLY_GRAPH --PE_graph PE_GRAPH
                   --binned BINNED [--max_iter MAX_ITER] [--thresh THRESH]
                   --output OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  --contigs CONTIGS     path to contigs file
  --assembler ASSEMBLER
                        assembler used (metaSPAdes or MEGAHIT)
  --assembly_graph ASSEMBLY_GRAPH
                        path to the .ag file
  --PE_graph PE_GRAPH   path to the .pe file
  --binned BINNED       path to the .csv file as initial binning
  --max_iter MAX_ITER   max iteration (default 100)
  --thresh THRESH       stop threshold (default 0.00000001)
  --output OUTPUT       output folder
```
In the OUTPUT folder, we provide two types of binning result:
- ```binning_result.csv```, each line is contig_name, cluster_id
- ```cluster.*.fasta```, the contigs in fasta format of each cluster

## Time and Memory comparing with GraphBin (v1.3)
- The comparison results can be accessed [here](https://drive.google.com/drive/folders/11U4YwiLLrcTCwpWy7Vax9n5Pk99E_8WL?usp=sharing).
- The machine used for comparison is CentOS 8.2 (64-bit), with Dual 26-core Intel Xeon Gold 6230R 2.10GHz CPU and 768GB RAM.
- The measured time and memory include GraphBin/METAMVGL binning on MaxBin2/MetaBAT2 initial binning results from metaSPAdes/MEGAHIT assembly on BMock12, SYNTH64 and Sharon datasets.
- The `time_memory/README.md` has the commands for binning, the evaluation results are in `time_memory/*/*/*.time` and generated by `time_memory/run_compare.sh`.