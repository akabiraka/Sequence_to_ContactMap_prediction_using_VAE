import numpy as np
import math
from Bio import SeqIO
import torch
import torch.nn.functional as F

import utility as Utility
import constants as CONSTANTS


class InputGenerator(object):
    """docstring for FeatureSelector."""

    def __init__(self):
        super(InputGenerator, self).__init__()
        self.window = CONSTANTS.WINDOW_SIZE
        self.stride = CONSTANTS.WINDOW_STRIDE

    def get_input_output_pre(self, pdb_code):
        """
        deprecated
            output format:
                [seq1, seq2, output_matrix]
                size of seq1, seq2 = 1 x l x 20
                output_matrix = l x l
        """
        one_hot_tensor = Utility.read_one_hot_tensor(pdb_code)
        contact_map_tensor = Utility.read_contact_map_tensor(pdb_code)
        print(pdb_code + ":",
              "1-hot size:", one_hot_tensor.shape,
              "contact-map size:", contact_map_tensor.shape)
        rows, cols = one_hot_tensor.shape
        half_width = math.floor(self.window / 2)
        a_input_output_set = []
        all_input_output_set = []
        for i in range(half_width, rows - half_width, self.stride):
            sub_seq1 = one_hot_tensor[i - half_width:i + half_width]
            sub_seq1 = sub_seq1.unsqueeze(0)
            sub_seq1 = sub_seq1.type(torch.float32)
            for j in range(half_width, rows - half_width, self.stride):
                # print(i - half_width, i + half_width,
                #       j - half_width, j + half_width)
                sub_seq2 = one_hot_tensor[j - half_width:j + half_width]
                sub_seq2 = sub_seq2.unsqueeze(0)
                sub_seq2 = sub_seq2.type(torch.float32)
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

    def get_input_output(self, pdb_code):
        """
            output format:
                input: concatenated sub_seq1, sub_seq2, [1 x 2n x self.window],
                        n=20 for AA letters
                out:

                size of seq1, seq2 = 1 x l x 20
                output_matrix = l x l
        """
        one_hot_tensor = Utility.read_one_hot_tensor(pdb_code)
        contact_map_tensor = Utility.read_contact_map_tensor(pdb_code)
        print(pdb_code + ":",
              "1-hot size:", one_hot_tensor.shape,
              "contact-map size:", contact_map_tensor.shape)
        rows, cols = one_hot_tensor.shape
        half_width = math.floor(self.window / 2)
        a_input_output_set = []
        all_input_output_set = []
        for i in range(half_width, rows - half_width, self.stride):
            sub_seq1 = self.format_seq(one_hot_tensor, i -
                                       half_width, i + half_width)
            for j in range(half_width, rows - half_width, self.stride):
                # print(i - half_width, i + half_width,
                #       j - half_width, j + half_width)
                sub_seq2 = self.format_seq(one_hot_tensor, j -
                                           half_width, j + half_width)
                input = torch.cat((sub_seq1, sub_seq2), 1)
                input.unsqueeze_(0)
                # print(input.size())
                out = contact_map_tensor[i - half_width:i +
                                         half_width, j - half_width:j + half_width]

                out = out.type(torch.float32)
                a_input_output_set = [input, out]
                all_input_output_set.append(a_input_output_set)
            #     break
            # break
        return all_input_output_set

    def format_seq(self, tensor, from_seq_index, to_seq_index):
        sub_seq = tensor[from_seq_index:to_seq_index]
        sub_seq.transpose_(0, 1)
        sub_seq = sub_seq.type(torch.float32)
        # print(sub_seq.size())
        return sub_seq
