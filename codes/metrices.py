import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sklearn.metrics as sk_metrics
from progress.bar import Bar

from protein_dataset import ProteinDataset
import constants as CONSTANTS
import utility as Utility
from models.basic_vae_1 import BasicVAE1


class Metrices(object):
    """docstring for FeatureSelector."""

    def __init__(self, model_path="../output_models/best_model_16.pth"):
        super(Metrices, self).__init__()
        self.window = CONSTANTS.WINDOW_SIZE
        self.stride = CONSTANTS.WINDOW_STRIDE
        self.save_plt = True

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model(model_path)

    def load_model(self, model_path):
        self.model = BasicVAE1()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def format_seq(self, tensor, from_seq_index, to_seq_index):
        sub_seq = tensor[from_seq_index:to_seq_index]
        sub_seq.transpose_(0, 1)
        sub_seq = sub_seq.type(torch.float32)
        # print(sub_seq.size())
        return sub_seq

    def compute_predicted_contact_map(self, pdb_code, model_no=0):
        """
        """
        one_hot_tensor = Utility.read_one_hot_tensor(pdb_code)
        contact_map_tensor = Utility.read_contact_map_tensor(pdb_code)
        result_contact_map_tensor = torch.zeros(
            contact_map_tensor.size(), dtype=torch.uint8)

        # print(result_contact_map_tensor.size(), contact_map_tensor.size())
        # print(pdb_code + ":",
        #       "1-hot size:", one_hot_tensor.shape,
        #       "contact-map size:", contact_map_tensor.shape)
        rows, cols = one_hot_tensor.shape
        half_width = math.floor(self.window / 2)
        a_input_output_set = []
        all_input_output_set = []
        for i in range(half_width, rows - half_width, self.stride):
            s1_from_idx = i - half_width
            s1_to_idx = i + half_width
            sub_seq1 = self.format_seq(one_hot_tensor, s1_from_idx, s1_to_idx)
            for j in range(half_width, rows - half_width, self.stride):
                s2_from_idx = j - half_width
                s2_to_idx = j + half_width
                sub_seq2 = self.format_seq(
                    one_hot_tensor, s2_from_idx, s2_to_idx)

                x = torch.cat((sub_seq1, sub_seq2), 1)
                x.unsqueeze_(0)
                # print(input.size())
                y = contact_map_tensor[s1_from_idx:s1_to_idx,
                                       s2_from_idx:s2_to_idx]

                y = y.type(torch.float32)
                x.unsqueeze_(0), y.unsqueeze_(0)
                x, y = x.to(self.device), y.to(self.device)
                y_prime, mu, logvar = self.model(x)
                # print(x.shape, y.shape, y_prime.shape)
                self.compute_result_contact_map(
                    result_contact_map_tensor, y_prime, s1_from_idx, s1_to_idx, s2_from_idx, s2_to_idx)
        # Utility.plot_images(
        #     [result_contact_map_tensor, contact_map_tensor], pdb_code + str(model_no), cols=2, save_plt=self.save_plt)
        return result_contact_map_tensor, contact_map_tensor

    def compute_result_contact_map(self, contact_map, y_prime, s1_from_idx, s1_to_idx, s2_from_idx, s2_to_idx):
        y_prime = y_prime.squeeze_(0).squeeze_(0).cpu()
        # print(y_prime.size(), s1_from_idx,
        #       s1_to_idx, s2_from_idx, s2_to_idx)
        y_prime_mask = torch.where(
            y_prime > .00008, torch.tensor(1), torch.tensor(0)).to(dtype=torch.uint8)

        # the equivalent of next two lines are at the end of this file
        contact_map[s1_from_idx:s1_to_idx, s2_from_idx:s2_to_idx] = contact_map[s1_from_idx:s1_to_idx,
                                                                                s2_from_idx:s2_to_idx] | y_prime_mask
        # contact_map[s1_from_idx:s1_to_idx,
        #             s2_from_idx:s2_to_idx] = y_prime_mask
        # plt.imshow(y_prime_mask)
        # plt.show()
        # plt.imshow(contact_map)
        # plt.show()

    def compute_metrices(self, y, y_prime):
        y, y_prime = y.view(-1), y_prime.view(-1)
        precision, recall, f_score, support = sk_metrics.precision_recall_fscore_support(
            y, y_prime, average='binary')
        # print(precision, recall, f_score, support)
        fpr, tpr, thresholds = sk_metrics.roc_curve(y, y_prime)
        auc_score = sk_metrics.auc(fpr, tpr)
        # print(fpr, tpr, thresholds)
        # print("\n\n")

        return precision, recall, f_score, fpr, tpr, thresholds, auc_score


# use 14 or 16
models = [14, 16]  # theses doing nothing:, 15, 17, 24, 25, 26]
for model_no in models:
    metrices = Metrices(
        model_path="../output_models/best_model_{}.pth".format(model_no))
    # pdbs = ["1kpgA", "4mt8A", "5cxxA", "2b9dA", "4a1rA",
    #         "1j3mA", "2zsiB", "5it6A", "6ji6A", "5ky4B",
    #         "4inkA", "6dewA", "5xauC", "5cegA", "5h0jA"]
    pdbs = Utility.get_pdb_identifiers(CONSTANTS.TEST_FILE)
    all_precisions = []
    all_recalls = []
    all_f_scores = []

    plt.figure(figsize=(15, 15))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    bar = Bar('Processing proteins:', max=len(pdbs))
    for i, pdb in enumerate(pdbs):
        # print("executing {} ... ...".format(pdb))
        predicted_cMap, real_cMap = metrices.compute_predicted_contact_map(
            pdb, model_no)
        precision, recall, f_score, fpr, tpr, thresholds, auc_score = metrices.compute_metrices(
            real_cMap, predicted_cMap)
        all_precisions.append(precision), all_recalls.append(
            recall), all_f_scores.append(f_score)

        if i < 30:
            plt.plot(
                fpr, tpr, label='{}.ROC-{}(AUC = {:.2f})'.format(i + 1, pdb, auc_score))
            plt.legend(loc='lower right')

        bar.next()
    bar.finish()

    plt.title("ROC Curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.show()
    plt.savefig("../output_images/roc_curve_model_{}.png".format(model_no))

    avg_precision, avg_recall, avg_f_score = np.mean(
        all_precisions), np.mean(recall), np.mean(f_score)
    print("Model: {}, precision: {}, recall: {}, f_score: {}".format(
        model_no, avg_precision, avg_recall, avg_f_score))


# import torch
# import numpy as np
# # r = torch.rand((3, 3))
# r = torch.tensor([[0.7506, 0.3832, 0.2267],
#         [0.4251, 0.9442, 0.1337],
#         [0.6699, 0.0650, 0.0047],
#                  [0.7506, 0.3832, 0.2267]])
# r_mask = r > .5
# x = torch.where(r > .5, torch.tensor(1), torch.tensor(0)).to(dtype=torch.uint8)
# # x = x.to(dtype=torch.uint8)
# print(r, r_mask)
# print(x)
# y = torch.zeros((10, 10), dtype=torch.uint8)
# print(y.dtype, x.dtype)
# y[0:4, 0:3] = y[0:4, 0:3] | x
# print(y)
# y[0:4:, 1:4]
# print(y)
