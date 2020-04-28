import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from input_generator import InputGenerator
import utility as Utility


class ProteinDataset(Dataset):
    """docstring for ProteinDataset."""

    def __init__(self, file):
        super(ProteinDataset, self).__init__()
        pdb_identifies_file = file  # '../inputs/train.txt'  # ../inputs/pdb_identifiers.txt'
        self.pdb_identifiers = Utility.get_pdb_identifiers(pdb_identifies_file)
        print(len(self.pdb_identifiers), " proteins in hand")
        self.input_generator = InputGenerator()
        self.records = self.generate_input_output_sets()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        # x, y = self.records[i]
        # print("from dataset: ", "x:", x.shape, "y:", y.shape)
        return self.records[i]

    def n_proteins(self):
        return len(self.pdb_identifiers)

    def generate_input_output_sets(self):
        records = []
        for identifier in self.pdb_identifiers:
            pdb_code = identifier
            inp_out_pairs = self.input_generator.get_input_output(pdb_code)
            # print(pdb_code, ":", len(inp_out_pairs))
            records.extend(inp_out_pairs)

        # print(len(records))
        return records
