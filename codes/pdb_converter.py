import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

from Bio.PDB import *
from Bio import SeqIO

import constants as CONSTANTS


class PDBConverter(object):
    """docstring for PDBConverter."""

    def __init__(self):
        super(PDBConverter, self).__init__()

        self.pdb_identifies_file = '../inputs/pdb_identifiers.txt'
        self.pdb_identifiers = self.get_pdb_identifiers(
            self.pdb_identifies_file)
        self.pdb_code = '2blq'
        self.threshhold = 12.0

        self.pdbl = PDBList()
        self.parser = MMCIFParser()

    def do(self):
        self.download()
        self.convert_into_distmatrices_contactmaps()
        self.convert_into_fasta()

    def download(self):
        for identifier in self.pdb_identifiers:
            self.pdb_code = identifier
            self.pdbl.retrieve_pdb_file(
                self.pdb_code, pdir=CONSTANTS.PDB_DIR, file_format=CONSTANTS.CIF)

    def convert_into_distmatrices_contactmaps(self):
        for identifier in self.pdb_identifiers:
            self.pdb_code = identifier
            pdb_filename = CONSTANTS.PDB_DIR + self.pdb_code + CONSTANTS.CIF_EXT
            structure = self.parser.get_structure(
                self.pdb_code, pdb_filename)
            all_residues = structure.get_residues()
            aa_residues, non_aa_residues = self.filter_aa_residues(
                all_residues)
            print("======================: ", len(aa_residues))
            dist_matrix = self.compute_distance_matrix(
                aa_residues, aa_residues)
            contact_map = np.where(dist_matrix < self.threshhold, 1, 0)
            filename = self.pdb_code + "_" + str(0) + CONSTANTS.CSV_EXT
            self.save(dist_matrix, filename,
                      CONSTANTS.DISTANCE_MATRIX_DIR)
            self.save(contact_map, filename, CONSTANTS.CONTACT_MAP_DIR)
            self.view(filename)

    def convert_into_fasta(self):
        for identifier in self.pdb_identifiers:
            self.pdb_code = identifier
            from_cif = CONSTANTS.PDB_DIR + self.pdb_code + CONSTANTS.CIF_EXT
            to_fasta = CONSTANTS.FASTA_DIR + self.pdb_code + CONSTANTS.FASTA_EXT
            records = SeqIO.parse(from_cif, "cif-atom")
            count = SeqIO.write(records, to_fasta, "fasta")

    def compute_res_res_distance(self, residue_1, residue_2):
        diff_vector = residue_1["CA"].coord - residue_2["CA"].coord
        return np.sqrt(np.sum(diff_vector * diff_vector))

    def compute_distance_matrix(self, chain_1, chain_2):
        dist_matrix = np.zeros((len(chain_1), len(chain_2)), np.float)
        for row, residue_1 in enumerate(chain_1):
            for col, residue_2 in enumerate(chain_2):
                dist_matrix[row, col] = self.compute_res_res_distance(
                    residue_1, residue_2)
        return dist_matrix

    def filter_aa_residues(self, chain):
        """
        a chain can be heteroatoms(water, ions, etc; anything that isn't an amino acid or nucleic acid)
        so this function get rid of atoms excepts amino-acids
        """
        aa_residues = []
        non_aa_residues = []
        non_aa = []
        for i in chain:
            if i.get_resname() in standard_aa_names:
                aa_residues.append(i)
            else:
                non_aa.append(i.get_resname())
                non_aa_residues.append(i)
        return aa_residues, non_aa_residues

    def save(self, matrix, filename, path):
        np.savetxt(path + filename, matrix, delimiter=',')

    def read(self, filename, path):
        return np.loadtxt(path + filename, delimiter=',')

    def get_pdb_identifiers(self, file):
        file_content = open(file).read()
        file_content = file_content.lower()
        return file_content.split()

    def view(self, filename):
        img1 = self.read(filename, CONSTANTS.DISTANCE_MATRIX_DIR)
        img2 = self.read(filename, CONSTANTS.CONTACT_MAP_DIR)
        images = [img1, img2]
        titles = ["distance matrix", "contact map"]
        self.__plot_images(images, filename, titles, cols=2)

    def __plot_images(self, images, img_name, titles=None, cols=3):
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
        # plt.savefig("output_images/{}.png".format(img_name))
        # plt.close()

    # def compute_dist_matrices(self, model, th=12.0):
    #     """
    #     th: threshhold
    #         default: 12 angstroms
    #     """
    #     dist_matrices = []
    #     contact_maps = []
    #     for chain_1 in model:
    #         chain_1_aa_residues, _ = self.filter_aa_residues(chain_1)
    #         for chain_2 in model:
    #             chain_2_aa_residues, _ = self.filter_aa_residues(chain_2)
    #             dist_matrix = self.compute_chain_chain_distance_matrix(
    #                 chain_1_aa_residues, chain_2_aa_residues)
    #             contact_map = np.where(dist_matrix < th, 1, 0)
    #             dist_matrices.append(dist_matrix)
    #             contact_maps.append(contact_map)
    #
    #     return dist_matrices, contact_maps
