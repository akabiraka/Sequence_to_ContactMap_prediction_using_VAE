import numpy as np
import math
from Bio import SeqIO
import torch
import torch.nn.functional as F

import utility as Utility
import constants as CONSTANTS


class InputGenerator(object):
    """docstring for FeatureSelector."""

    def __init__(self, window=30, stride=30):
        super(InputGenerator, self).__init__()
        self.window = window
        self.stride = stride

    def get_input_output(self, pdb_code):
        one_hot_tensor = Utility.read_one_hot_tensor(pdb_code)
        contact_map_tensor = Utility.read_contact_map_tensor(pdb_code)
        print(pdb_code + ": ", one_hot_tensor.shape, contact_map_tensor.shape)
        rows, cols = one_hot_tensor.shape
        half_width = math.floor(self.window / 2)
        a_input_output_set = []
        all_input_output_set = []
        for i in range(half_width, rows - half_width, self.stride):
            sub_seq1 = one_hot_tensor[i - half_width:i + half_width]
            for j in range(half_width, rows - half_width, self.stride):
                # print(i - half_width, i + half_width,
                #       j - half_width, j + half_width)
                sub_seq2 = one_hot_tensor[j - half_width:j + half_width]
                out = contact_map_tensor[i - half_width:i +
                                         half_width, j - half_width:j + half_width]
                a_input_output_set = [sub_seq1, sub_seq2, out]
                all_input_output_set.append(a_input_output_set)

        # here is how to manipulate all_input_output_set
        # all_outputs = []
        # for i, data in enumerate(all_input_output_set):
        #     sub_seq1, sub_seq2, out = data
        #     all_outputs.append(out)
        #     print(sub_seq1.shape, sub_seq2.shape, out.shape)
        # Utility.plot_images(all_outputs, img_name=pdb_code, cols=5)
        return all_input_output_set
