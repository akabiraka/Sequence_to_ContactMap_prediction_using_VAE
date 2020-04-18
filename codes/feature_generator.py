import numpy as np
from Bio import SeqIO
import torch
import torch.nn.functional as F

import utility as Utility
import constants as CONSTANTS


class FeatureGenerator(object):
    """docstring for FeatureGenerator."""

    def __init__(self, file):
        super(FeatureGenerator, self).__init__()

        pdb_identifies_file = file
        self.pdb_identifiers = Utility.get_pdb_identifiers(pdb_identifies_file)
        self.AALetters = CONSTANTS.AMINO_ACID_20
        self.n_aaletters = len(self.AALetters)
        self.AA_c2i_dict, self.AA_i2c_dict = self.__get_amino_acid_seq_dict()

    def one_hot(self):
        """
        Generate one-hot tensor feature for each pdb id.
        size of each feature: [seq_len x n_amino_acid], i.e. [100 x 20]
        """
        for pdb_code in self.pdb_identifiers:
            # filename = CONSTANTS.FASTA_DIR + pdb_code + CONSTANTS.FASTA_EXT
            # file = open(filename)
            # records = SeqIO.parse(file, CONSTANTS.FASTA)
            # seq = self.__get_sequence(records)
            seq = Utility.read_fasta_seq(pdb_code)
            numeric = self.__seq_2_numeric(seq)
            one_hot_tensor = F.one_hot(torch.tensor(numeric),
                                       num_classes=self.n_aaletters)
            print(pdb_code + ": ", one_hot_tensor.shape)
            Utility.save_one_hot_tensor(one_hot_tensor, pdb_code)

    def __get_sequence(self, records):
        """
        returns all amino acid sequence by concatenating all chains
        """
        seq = ""
        for record in records:
            print(record.seq)
            seq += record.seq
        return seq

    def __get_amino_acid_seq_dict(self):
        """
        returns two directories, char2Int and int2Char mapping
        """
        char2Int = {}
        int2Char = {}
        for i, char in enumerate(self.AALetters):
            char2Int[char] = i
            int2Char[i] = char
        return char2Int, int2Char

    def __seq_2_numeric(self, seq):
        """
        returns seq to numeric mapping
        """
        numeric = []
        for i, letter in enumerate(seq):
            if letter not in self.AA_c2i_dict:
                # print("'{}' does not exist in AMINO_ACID_letters".format(letter))
                continue
            numeric.append(self.AA_c2i_dict.get(letter))
        return numeric
