# -*- coding:utf-8 -*-
#!/usr/bin/env python3
from re import S
import sys
import time
import argparse
import os
import logging

from GCNbin_utils import feature_utils
from GCNbin_utils import marker_gene_utils
from GCNbin_utils import graph_utils
from GCNbin_utils import sdcn_utils
from GCNbin_utils.pretrain_ae import pretrain_ae


# Setup argument parser
# ---------------------------------------------------

ap = argparse.ArgumentParser(description="""GCNbin is a multi-view and deep learning-based metagenomic contigs binning tool, GCNbin performs synchronous feature learning for compositional values and structural graphs by autoencoder and graph convolutional network respectively, and uses a dual self-supervised module to unify the results of the two neural networks and to guide the binning task. """)

ap.add_argument("--assembler", required=True,
                help="name of the assembler used")
ap.add_argument("--contigs", required=True, help="path to the contigs file")
ap.add_argument("--graph", required=True,
                help="path to the assembly graph file")
ap.add_argument("--output", required=True, help="path to the output folder")
ap.add_argument("--paths", required=False,
                help="path to the contigs.paths file")
ap.add_argument("--abundance", required=False,default= None,
                help="path to the abundance file")

ap.add_argument("--prefix", required=False, default='meta',
                help="prefix for the output file")
ap.add_argument("--bam", required=True,
                help="path to the bam file")
ap.add_argument("--min_length", required=False,default= 300,
                help="cluster_K")
ap.add_argument("--K", required=False,type=int,default=0,
                help="cluster_K")
ap.add_argument("--lambda", required=False,type=int,default=0,
                help="cluster_K,[default: auto]")
ap.add_argument("--topk", required=False, type=int, default=1,
                help="K number of KNN graph ")
ap.add_argument("--learn_rate", required=False, type=float, default=1e-3 ,
                help="learn rate [default: 1e-3]")
ap.add_argument("--h1", required=False, type=int, default=100)
ap.add_argument("--h2", required=False, type=int, default=70)
ap.add_argument("--z", required=False, type=int, default=50)
ap.add_argument("--nthreads", required=False, type=int, default=8,
                help="number of threads to use. [default: 8]")

# Parse arguments
args = vars(ap.parse_args())
assembler = args["assembler"]
contigs_file = args["contigs"]
assembly_graph_file = args["graph"]
contig_paths_file = args["paths"]
bam_file = args["bam"]
abundance_file = args["abundance"]
output_path = args["output"]
prefix = args["prefix"]
topk = args["topk"]
lr= args["learn_rate"]
n_h1 = args["h1"]
n_h2 = args["h2"]
n_z = args["z"]
cluster_K =args["K"]
nthreads = args["nthreads"]
min_length = args["min_length"]
lam= args["lambda"]

# Setup logger
# -----------------------

logger = logging.getLogger('GCNbin 1.0')
logger.setLevel(logging.DEBUG)
logging.captureWarnings(True)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
consoleHeader = logging.StreamHandler()
consoleHeader.setFormatter(formatter)
consoleHeader.setLevel(logging.INFO)
logger.addHandler(consoleHeader)

# Setup output path for log file
fileHandler = logging.FileHandler(output_path + "/" + prefix + "GCNbin.log")
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

logger.info(
    "Welcome to GCNbin.")

logger.info("Input arguments: ")
logger.info("Assembler used: " + assembler)
logger.info("Contigs file: " + contigs_file)
logger.info("Assembly graph file: " + assembly_graph_file)

logger.info("Final binning output file: " + output_path)

logger.info("Number of threads: " + str(nthreads))

logger.info("GCNbin started")

start_time = time.time()


# Get links of the assembly graph
# ------------------------------------------------------------------------


node, sequences, coverages, n_samples, n_contigs,length_list,contig_name_list = feature_utils.get_seq_and_cov(contigs_file, abundance_file, min_length, output_path, prefix, assembler)
logger.info("Total number of contigs is " + str(n_contigs))
# Construct the assembly graph
# -------------------------------
normalized_tetramer_profiles = feature_utils.get_tetramer_profiles(output_path, prefix, sequences, nthreads)
n_input = n_samples +136
logger.info("Pretarin the AE." )
z_contigs = pretrain_ae(n_h1 , n_h2 , n_z, output_path, prefix, n_input,n_samples)

AG_graphURL = output_path+prefix+".ag"
PE_graphURL = output_path+prefix+".pe"
Seq_graphURL = output_path+prefix+"_seq.txt"
if os.path.isfile(AG_graphURL) and os.path.isfile(PE_graphURL):
   logger.info("AG and PE graphs are already exist.")
   
else:
    logger.info("Constructing AG and PE graphs.")
    try:
    # Use the script in METAMVGL
       if assembler == "spades":
          graph_utils.get_AGandPE_graph_spades(bam_file, assembly_graph_file, contig_paths_file, prefix, output_path)

       if assembler == "megahit":
          graph_utils.get_AGandPE_graph_megahit(bam_file, assembly_graph_file, contigs_file, prefix, output_path)
    except:
       logger.error("Fail to constructing AG and PE graphs.")
       logger.info("Exiting GCNbin... Bye...!")
       sys.exit(1)
graph_utils.construct_multi_graph(output_path,prefix,node)
logger.info("Constructing KNN graphs.")
if os.path.isfile(Seq_graphURL):
   logger.info("KNN graphs are already exist.")
   
elif topk ==0:
    logger.info("Don't use KNN graphs.")
   
else:
   try:
        graph_utils.construct_seq_graph(output_path,normalized_tetramer_profiles,coverages,topk,prefix,n_contigs,nthreads)
   except:
       logger.error("Fail to constructing KNN graphs. ")
       logger.info("Exiting GCNbin... Bye...!")
       sys.exit(1)

# Get contigs with marker genes
# -----------------------------------------------------
if cluster_K == 0:
   logger.info("Scanning for single-copy marker genes")
   cluster_K = marker_gene_utils.scan_for_marker_genes(contigs_file, z_contigs,nthreads, bestK=0)
   logger.info("binning with cluster_K:"+str(cluster_K))
else:

   logger.info("binning with cluster_K:"+str(cluster_K))


y_pred,epoch,gailv= sdcn_utils.train_sdcn(n_h1, n_h2, n_z, cluster_K, n_input, prefix, output_path, lr,length_list ,nthreads, n_samples,node)
gailv1=gailv.detach().numpy()
# Write binning result.
# -----------------------------------------------------
logger.info("Write binning result.")
if lam == 0:
    output_all_contigs = open(output_path+prefix+"_label.csv", 'w',encoding="utf-8")
    for n in range(n_contigs):
            output_all_contigs.write(str(y_pred[n])+","+str(contig_name_list    [n])+'\n')
    
    if cluster_K < 10:
        gailv_c= 2/cluster_K
    else:
        gailv_c= 3/cluster_K
    output_pure_contigs = open(output_path+prefix+"label_pure.csv", 'w',encoding="utf-8")
    for n in range(n_contigs):
        if max(gailv1[n])>gailv_c:
                    output_pure_contigs.write(str(y_pred[n])+","+str(contig_name_list[n])+'\n')
else:
    gailv_c= (1+lam)/cluster_K
    output_pure_contigs = open(output_path+prefix+"label_pure.csv", 'w',encoding="utf-8")
    for n in range(n_contigs):
        if max(gailv1[n])>gailv_c:
                    output_pure_contigs.write(str(y_pred[n])+","+str(contig_name_list[n])+'\n')





logger.info("Thank you for using GCNbin!")
