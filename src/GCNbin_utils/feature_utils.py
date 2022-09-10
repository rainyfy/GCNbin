#!/usr/bin/env python3

import numpy as np
import itertools
import os
import pickle
import logging

from Bio import SeqIO
from multiprocessing import Pool


# Create logger
#logger = logging.getLogger('GCNbin 1.0')

# Set complements of each nucleotide
complements = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

# Set bits for each nucleotide
nt_bits = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

VERY_SMALL_VAL = 0.0001


def get_rc(seq):
    rev = reversed(seq)
    return "".join([complements.get(i, i) for i in rev])


def mer2bits(kmer):
    bit_mer = nt_bits[kmer[0]]
    for c in kmer[1:]:
        bit_mer = (bit_mer << 2) | nt_bits[c]
    return bit_mer


def compute_kmer_inds(k):
    kmer_inds = {}
    kmer_count_len = 0

    alphabet = 'ACGT'

    all_kmers = [''.join(kmer)
                 for kmer in itertools.product(alphabet, repeat=k)]
    all_kmers.sort()
    ind = 0
    for kmer in all_kmers:
        bit_mer = mer2bits(kmer)
        rc_bit_mer = mer2bits(get_rc(kmer))
        if rc_bit_mer in kmer_inds:
            kmer_inds[bit_mer] = kmer_inds[rc_bit_mer]
        else:
            kmer_inds[bit_mer] = ind
            kmer_count_len += 1
            ind += 1

    return kmer_inds, kmer_count_len


def count_kmers(args):
    seq, k, kmer_inds, kmer_count_len = args
    profile = np.zeros(kmer_count_len)
    arrs = []
    seq = list(seq.strip())

    for i in range(0, len(seq) - k + 1):
        bit_mer = mer2bits(seq[i:(i + k)])
        index = kmer_inds[bit_mer]
        profile[index] += 1

    return profile/max(1, sum(profile)),(profile-np.min(profile))/(np.max(profile)-np.min(profile))


def get_tetramer_profiles(output_path, prefix, sequences, nthreads):
    tetramer_profiles = {}
    normalized_tetramer_profiles = {}
    normalized_tetramer_profiles2 = {}
    

    if os.path.isfile(output_path + prefix +"_normalized_contig_tetramers2.pickle"):

        with open(output_path + prefix +'_normalized_contig_tetramers2.pickle', 'rb') as handle:
            normalized_tetramer_profiles2 = pickle.load(handle)

        i = 0
        with open(output_path + prefix +"_normalized_contig_tetramers2.txt") as tetramers_file:
            for line in tetramers_file.readlines():
                f_list = np.array([float(i)
                                for i in line.split(" ") if i.strip()])
                normalized_tetramer_profiles2[i] = f_list
                i += 1
#
    else:
        
        kmer_inds_4, kmer_count_len_4 = compute_kmer_inds(4)

        pool = Pool(nthreads)
        record_tetramers = pool.map(
            count_kmers, [(seq, 4, kmer_inds_4, kmer_count_len_4) for seq in sequences])
        pool.close()

        normalized1 = [x[1] for x in record_tetramers]
        normalized2 = [x[0] for x in record_tetramers]
        # unnormalized = [x[0] for x in record_tetramers]

        i = 0

        # for l in range(len(unnormalized)):
        #     tetramer_profiles[i] = unnormalized[l]
        #     i += 1

        # with open(output_path + "contig_tetramers.txt", "w+") as myfile:
        #     for l in range(len(unnormalized)):
        #         for j in range(len(unnormalized[l])):
        #             myfile.write(str(unnormalized[l][j]) + " ")
        #         myfile.write("\n")
        for l in range(len(normalized2)):
            normalized_tetramer_profiles2[i] = normalized2[l]
            i += 1
        i = 0

        for l in range(len(normalized1)):
            normalized_tetramer_profiles[i] = normalized1[l]
            i += 1

        with open(output_path + prefix +'_normalized_contig_tetramers.pickle', 'wb') as handle:
            pickle.dump(normalized_tetramer_profiles, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(output_path + prefix +'_normalized_contig_tetramers2.pickle', 'wb') as handle:
            pickle.dump(normalized_tetramer_profiles2, handle, protocol=pickle.HIGHEST_PROTOCOL)


        with open(output_path + prefix +"_normalized_contig_tetramers.txt", "w+") as myfile:
             for l in range(len(normalized1)):
                for j in range(len(normalized1[l])):
                   myfile.write(str(normalized1[l][j]) + " ")
                myfile.write("\n")

        with open(output_path + prefix +"_normalized_contig_tetramers2.txt", "w+") as myfile2:
             for l in range(len(normalized2)):
                for j in range(len(normalized2[l])):
                   myfile2.write(str(normalized2[l][j]) + " ")
                myfile2.write("\n")    
    return normalized_tetramer_profiles2

def get_seq_and_cov(contigs_file, abundance_file, min_length, output_path, prefix, assembler):

    coverages = {}

    contig_lengths = {}
    
    length_list=[]
        
    node = {}
    
    contig_num = {}
    
    j = 0
    
    contig_name_list = []
    sequences = []
    
    if assembler == "megahit":
        for index, record in enumerate(SeqIO.parse(contigs_file, "fasta")):
            length = len(record.seq)
            
            contig_lengths[record.name]=length
            min_length=int(min_length)
            
            if length >= min_length:
                
                node_name=str(record.name)
                node_name=node_name.split(' ')
                contig_name_list.append(node_name)
                node_name=node_name[0]
                length_list.append(length)
                node[node_name]=j
                contig_num[record.name]=j
                
                
                sequences.append(str(record.seq))
                
                j+=1
    else:
        for index, record in enumerate(SeqIO.parse(contigs_file, "fasta")):
    
            length = int(len(record.seq))
            contig_lengths[record.name]=length
            min_length=int(min_length)
            if length >= min_length:
          
                node_name=str(record.name)
                contig_name_list.append(node_name)
                node_name=node_name.split('_')
                node_name=node_name[0]+'_'+node_name[1]
                length_list.append(length)
                node[node_name]=j
                contig_num[record.name]=j
                sequences.append(str(record.seq))
                
                j+=1
    n_contigs = j
    if abundance_file !=None:
        with open(abundance_file, "r") as my_abundance:
             for line in my_abundance:
            
                strings = line.strip().split("\t")
                if contig_lengths[strings[0]] >= min_length:
                
                   num=contig_num[strings[0]]

                
                   for i in range(1, len(strings)):

                       contig_coverage = float(strings[i])

                       if contig_coverage < VERY_SMALL_VAL:
                          contig_coverage = VERY_SMALL_VAL
                       if num not in coverages:
                          coverages[num] = [contig_coverage]
                       else:
                          coverages[num].append(contig_coverage)
        n_samples = len(coverages[0])
        with open(output_path + prefix +"_normalized_coverages.txt", "w+") as myfile:
             for l in range(len(coverages)):
                 for j in range(len(coverages[l])):
                     myfile.write(str(coverages[l][j]) + " ")
                 myfile.write("\n")
    else:
         n_samples = 0
         coverages= None
            
    return node, sequences, coverages, n_samples, n_contigs, length_list,contig_name_list
    
def get_pre_seq(contigs_file, min_length):

    sequences = []
    

    for index, record in enumerate(SeqIO.parse(contigs_file, "fasta")):
        length = len(record.seq)


    return sequences
    
