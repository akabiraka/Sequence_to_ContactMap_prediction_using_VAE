import torch
import constants as CONSTANTS


def get_pdb_identifiers(file):
    file_content = open(file).read()
    file_content = file_content.lower()
    return file_content.split()


def save_tensor(tensor, path, filename):
    file = path + filename + CONSTANTS._ONE_HOT + CONSTANTS.PT_EXT
    torch.save(tensor, file)
