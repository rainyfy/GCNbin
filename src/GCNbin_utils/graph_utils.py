
#!/usr/bin/env python3

import os
import pathlib
import logging
import pandas as pd
import numpy as np
import h5py
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import sys
import math
import operator
import logging
from multiprocessing import Pool
from scipy.spatial import distance

# Constants set from MaxBin 2.0
MU_INTRA, SIGMA_INTRA = 0, 0.01037897 / 2
MU_INTER, SIGMA_INTER = 0.0676654, 0.03419337
VERY_SMALL_DOUBLE = 1e-10

# create logger
logger = logging.getLogger('GCNbin 1.0')


# Modified from SolidBin
def get_AGandPE_graph_spades(bam_file, assembly_graph_file, contig_paths_file, prefix, output_path):

    software_path = pathlib.Path(__file__).parent.parent.absolute()

    prep_graphURL = os.path.join(software_path.parent, 'auxiliary' , 'METAMVGL-main',
                            'prep_graph')
    logger.debug(prep_graphURL)
    try:
        prep_graphCmd = prep_graphURL+" -a "+"metaSPAdes"+" -p "+contig_paths_file+" -g "+assembly_graph_file+" -b "+bam_file+" -o "+output_path+prefix
        logger.debug("exec cmd: "+prep_graphCmd)
        status = os.system(prep_graphCmd)
        if status == 0:
           logger.debug("success")
    except:
        logger.debug("AG_PE_graph construct failed!")

def get_AGandPE_graph_megahit(bam_file, assembly_graph_file, contig_file, prefix, output_path):

    software_path = pathlib.Path(__file__).parent.parent.absolute()

    prep_graphURL = os.path.join(software_path.parent, 'auxiliary' , 'METAMVGL-main',
                              'prep_graph')
    logger.debug(prep_graphURL)

    try:
        prep_graphCmd = prep_graphURL+" -a "+"MEGAHIT"+" -c "+contig_file+" -g "+assembly_graph_file+" -b "+bam_file+" -o "+output_path+prefix
        logger.debug("exec cmd: "+prep_graphCmd)
        status = os.system(prep_graphCmd)
        if status == 0:
           logger.debug("success")
    except:
        logger.debug("AG_graph construct failed!")

def construct_multi_graph(output_path,prefix,node):
    AG_graphURL = output_path+prefix+".ag"
    PE_graphURL = output_path+prefix+".pe"

    if not os.path.isfile(AG_graphURL):
        print("AG_graph not exists!")
        logger.debug("AG_graph not exists!")
        sys.exit(1)

    if os.path.isfile(AG_graphURL):
        if not os.path.exists(PE_graphURL):
            print("AG_graph not exists!")
            logger.debug("PE_graph not exists!")
            sys.exit(1)
    fname1 = open(output_path + prefix + "_ag.txt",'w',encoding="utf-8")
    fname2 = open(output_path + prefix + "_pe.txt",'w',encoding="utf-8")
    graph1 = open(AG_graphURL, 'r')
    line = graph1.readline()
    while line != "":
        line = line.strip()
        strings = line[:-1].split()
        if line[-1] == ':':
            contig = strings[0]
        elif line[-1] == ';':
            if contig in node and strings[0] in node:
              fname1.write(str(node[contig])+' '+str(node[strings[0]])+'\n')
        line = graph1.readline()
    graph1.close()
    graph2 = open(PE_graphURL, 'r')
    line = graph2.readline()
    while line != "":
        line = line.strip()
        strings = line[:-1].split()
        if line[-1] == ':':
            contig = strings[0]
        elif line[-1] == ';':
            if contig in node and strings[0] in node:
              fname2.write(str(node[contig])+' '+str(node[strings[0]])+'\n')
        line = graph2.readline()
    graph2.close()

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = sd*(2*math.pi)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


def get_tetramer_distance(seq1, seq2):
    return distance.euclidean(seq1, seq2)


def get_coverage_distance(cov1, cov2):
    return distance.euclidean(cov1, cov2)


def get_comp_probability(tetramer_dist):
    gaus_intra = normpdf(tetramer_dist, MU_INTRA, SIGMA_INTRA)
    gaus_inter = normpdf(tetramer_dist, MU_INTER, SIGMA_INTER)
    if gaus_intra ==0:
       cp=0
    else:
       cp= gaus_intra/(gaus_intra+gaus_inter)
    return cp


def get_cov_probability(cov1, cov2):
    poisson_prod_1 = 1
    poisson_prod_2 = 1
    for i in range(len(cov1)):
        # Adapted fromhttp://www.masaer.com/2013/10/08/Implementing-Poisson-pmf.html
        poisson_pmf_1 = math.exp(
            (cov1[i] * math.log(cov2[i])) - math.lgamma(cov1[i] +1.0) - cov2[i])
        poisson_pmf_2 = math.exp(
            (cov2[i] * math.log(cov1[i])) - math.lgamma(cov2[i] +1.0) - cov1[i])
        if poisson_pmf_1 < VERY_SMALL_DOUBLE:
            poisson_pmf_1 = VERY_SMALL_DOUBLE
        if poisson_pmf_2 < VERY_SMALL_DOUBLE:
            poisson_pmf_2 = VERY_SMALL_DOUBLE
        poisson_prod_1 = poisson_prod_1 * poisson_pmf_1
        poisson_prod_2 = poisson_prod_2 * poisson_pmf_2
    return min(poisson_prod_1, poisson_prod_2)


def count_pro(args):
    i,normalized_tetramer_profiles,coverages,n_contigs,topk =args
    dist=[]
    for j in range(n_contigs):
        tetramer_dist =get_tetramer_distance(normalized_tetramer_profiles[i],normalized_tetramer_profiles[j])
        prob_comp = get_comp_probability(tetramer_dist)
        prob_cov = get_cov_probability(coverages[i],coverages[j])
        prob_product = prob_comp * prob_cov
#            prob_product = prob_comp
        log_prob = 0
        if prob_product > 0.0:
           log_prob = - \
               (math.log(prob_comp, 10) +
                   math.log(prob_cov, 10))
        else:
           log_prob=1000
        dist.append(log_prob)   
    ind = np.argpartition(dist, (topk+1))[:(topk+1)] 
    print(i)
    return ind

def count_pro_withoutcov(args):
    i,normalized_tetramer_profiles,n_contigs,topk =args
    dist=[]
    for j in range(n_contigs):
        tetramer_dist =get_tetramer_distance(normalized_tetramer_profiles[i],normalized_tetramer_profiles[j])
        prob_comp = get_comp_probability(tetramer_dist)
        log_prob = 0
        if prob_comp > 0.0:
           log_prob = - \
               (math.log(prob_comp, 10))
        else:
           log_prob=1000
        dist.append(log_prob)   
    ind = np.argpartition(dist, (topk+1))[:(topk+1)] 
    print(i)
    return ind



def construct_seq_graph(output_path,normalized_tetramer_profiles,coverages,topk,prefix,n_contigs,nthreads):
    inds=[]
    pool = Pool(nthreads)
    if coverages == None:
       inds = pool.map(
           count_pro_withoutcov, [(i, normalized_tetramer_profiles, n_contigs,topk) for i in range (n_contigs)])
    else:
       inds = pool.map(
           count_pro, [(i, normalized_tetramer_profiles,coverages,n_contigs,topk) for i in range (n_contigs)])
    pool.close()  
    fname = open(output_path + prefix + "_seq.txt",'w',encoding="utf-8")  
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                fname.write('{} {}\n'.format(i, vv))

    fname.close()

            


