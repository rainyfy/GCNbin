#!/usr/bin/env python3

import os
import pathlib
import logging
from sklearn.cluster import KMeans
import numpy as np
from Bio import SeqIO
import logging
import os
import argparse
import sys

import re
import csv
import time


# create logger
logger = logging.getLogger('GCNbin 1.0')

def silhouette(X, W, label):
    X_colsum = np.sum(X ** 2, axis=1)
    X_colsum = X_colsum.reshape(len(X_colsum), 1)
    W_colsum = np.sum(W ** 2, axis=1)
    W_colsum = W_colsum.reshape(len(W_colsum), 1)

    Dsquare = np.tile(X_colsum, (1, W.shape[0])) + np.tile(W_colsum.T, (X.shape[0], 1)) - 2 * X.dot(W.T)
    # avoid error caused by accuracy
    Dsquare[Dsquare < 0] = 0
    D = np.sqrt(Dsquare)
    aArr = D[np.arange(D.shape[0]), label]
    D[np.arange(D.shape[0]), label] = np.inf
    bArr = np.min(D, axis=1)
    tmp = (bArr - aArr) / np.maximum(aArr, bArr)
    return np.mean(tmp)

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
    
# Modified from SolidBin
def scan_for_marker_genes(contig_file, x_contigs, nthreads, bestK=0):
    
    software_path = pathlib.Path(__file__).parent.parent.absolute()

    fragScanURL = os.path.join(software_path.parent, 'auxiliary',
                               'FragGeneScan1.31', 'run_FragGeneScan.pl')
    hmmExeURL = os.path.join(software_path.parent, 'auxiliary', 'hmmer-3.3',
                             'src', 'hmmsearch')
    markerExeURL = os.path.join(software_path.parent, 'auxiliary', 'test_getmarker.pl')
    
    markerURL = os.path.join(software_path.parent, 'auxiliary', 'marker.hmm')

    logger.debug(markerURL)
    
    seedURL = contig_file + ".seed"
    fragResultURL = contig_file+".frag.faa"
    hmmResultURL = contig_file+".hmmout"
    
    if not (os.path.exists(fragResultURL)):
        fragCmd = fragScanURL+" -genome="+contig_file+" -out="+contig_file + \
            ".frag -complete=0 -train=complete -thread="+str(nthreads)+" 1>" + \
            contig_file+".frag.out 2>"+contig_file+".frag.err"
        logger.debug("exec cmd: "+fragCmd)
        os.system(fragCmd)

    if os.path.exists(fragResultURL):
        if not (os.path.exists(hmmResultURL)):
            hmmCmd = hmmExeURL+" --domtblout "+hmmResultURL+" --cut_tc --cpu "+str(nthreads)+" " + \
                markerURL+" "+fragResultURL+" 1>"+hmmResultURL+".out 2>"+hmmResultURL+".err"
            logger.debug("exec cmd: "+hmmCmd)
            os.system(hmmCmd)
        if os.path.exists(hmmResultURL):
            if not (os.path.exists(seedURL)):
                markerCmd = markerExeURL + " " + hmmResultURL + " " + contig_file + " 1000 " + seedURL
                logger.debug("exec cmd: " + markerCmd)
                os.system(markerCmd)

            if os.path.exists(seedURL):
                candK = file_len(seedURL)+2
                maxK = min(3 * candK,len(x_contigs))
                stepK = 2

            else:
                logger.info("seed not exist, k start from 3 " )
                candK = 3
                maxK = min(20,len(x_contigs))
                stepK = 2



        else:
            logger.debug("HMMER search failed! Path: " +
                         hmmResultURL + " does not exist.")
    else:
        logger.debug("FragGeneScan failed! Path: " +
                     fragResultURL + " does not exist.")
    X_mat=x_contigs.numpy()
    if bestK == 0:
        bestK = candK
        if candK==maxK:
            bestK = candK
            bestSilVal = 0
        else:
            bestSilVal = 0
            t = time.time()
            for k in range(candK, maxK, stepK):
                kmeans = KMeans(n_clusters=k, init='k-means++', random_state=9, n_jobs=-1)
                kmeans.fit(X_mat)
                silVal = silhouette(X_mat, kmeans.cluster_centers_, kmeans.labels_)
                logger.info("k:" + str(k) + "\tsilhouette:" + str(silVal) + "\telapsed time:" + str(time.time() - t))
                t = time.time()

                if silVal > bestSilVal:
                    bestSilVal = silVal
                    bestK = k
                else:
                    break
        
        #candKold=candK
        candK = bestK + candK
        if maxK > candK:
            bestSilVal_2nd = 0
            for k in range(candK, maxK, stepK):
                kmeans = KMeans(n_clusters=k, init='k-means++', random_state=9, n_jobs=-1)
                kmeans.fit(X_mat)
                silVal_2nd = silhouette(X_mat, kmeans.cluster_centers_, kmeans.labels_)
                logger.info("k:" + str(k) + "\tsilhouette:" + str(silVal_2nd) + "\telapsed time:" + str(time.time() - t))
                t = time.time()
                if silVal_2nd > bestSilVal_2nd:
                    bestSilVal_2nd = silVal_2nd
                    bestK_2nd = k
                else:
                    break
            if bestSilVal_2nd > bestSilVal:
               bestSilVal = bestSilVal_2nd
               bestK= bestK_2nd

        logger.info("bestk:" + str(bestK) + "\tsilVal:" + str(bestSilVal))


    return bestK
