import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import pickle

from constants import *
from Bio.PDB import *


def get_pdb_identifiers(file):
    file_content = open(file).read()
    file_content = file_content.lower()
    return file_content.split()


def save_tensor(tensor, file):
    torch.save(tensor, file)


def save_np_array(np_array, file):
    np.savetxt(file, np_array, delimiter=',')


def save_one_hot_tensor(tensor, pdb_code):
    file = FEATURE_DIR + pdb_code + ONE_HOT_ + PT_EXT
    save_tensor(tensor, file)


def save_contact_map(np_array, pdb_code):
    file = CONTACT_MAP_DIR + pdb_code
    save_np_array(np_array, file + CSV_EXT)
    save_tensor(torch.tensor(np_array), file + PT_EXT)


def save_distance_matrix(matrix, pdb_code):
    file = DISTANCE_MATRIX_DIR + pdb_code
    save_np_array(matrix, file + CSV_EXT)
    save_tensor(torch.tensor(matrix), file + PT_EXT)


def read_tensor(path, filename):
    return torch.load(path + filename)


def read_one_hot_tensor(pdb_code):
    filename = pdb_code + ONE_HOT_ + PT_EXT
    return read_tensor(FEATURE_DIR, filename)


def read_contact_map_tensor(pdb_code):
    filename = pdb_code + PT_EXT
    return read_tensor(CONTACT_MAP_DIR, filename)


def plot_images(images, img_name, titles=None, cols=3):
    rows = math.ceil(len(images) / cols)
    for i in range(len(images)):
        index = i + 1
        plt.subplot(rows, cols, index)
        plt.imshow(images[i])
        if titles is not None:
            plt.title(img_name + ": " + str(titles[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def save_itemlist(itemlist, file):
    with open(file, 'w') as f:
        for item in itemlist:
            f.write("%s\n" % item)


def write_to_log(data):
    with open(LOG_FILE, "a") as log_file:
        log_file.writelines(data)


def save_fasta_seq(seq, pdb_code):
    filename = FASTA_DIR + pdb_code + FASTA_EXT
    with open(filename, "w") as f:
        f.write(seq)


def read_fasta_seq(pdb_code):
    filename = FASTA_DIR + pdb_code + FASTA_EXT
    with open(filename, "r+") as f:
        return f.read()
