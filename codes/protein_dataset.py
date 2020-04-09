import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from input_generator import InputGenerator
import utility as Utility


class ProteinDataset(Dataset):
    """docstring for ProteinDataset."""

    def __init__(self):
        super(ProteinDataset, self).__init__()
        pdb_identifies_file = '../inputs/pdb_identifiers.txt'
        self.pdb_identifiers = Utility.get_pdb_identifiers(pdb_identifies_file)
        self.input_generator = InputGenerator()

    def __len__(self):
        return len(self.pdb_identifiers)

    def __getitem__(self, i):
        pdb_code = self.pdb_identifiers[i]
        all_input_output_set = self.input_generator.get_input_output(pdb_code)
        return all_input_output_set
