import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

from Bio.PDB import *
from Bio import SeqIO

import constants as CONSTANTS
import utility as Utility


class PDBConverter(object):
    """docstring for PDBConverter."""

    def __init__(self, file):
        super(PDBConverter, self).__init__()

        pdb_identifies_file = file
        self.pdb_identifiers = self.get_pdb_identifiers(pdb_identifies_file)
        self.pdb_code = '2blq'
        self.threshhold = 12.0

        self.min_len = 32
        self.max_len = 300
        self.keep_n_pdbs = 1000
        self.view_every = 20

        self.pdbl = PDBList()
        self.parser = MMCIFParser()
        self.aa_3to1 = CONSTANTS.AMINO_ACID_3TO1

    def do(self):
        self.download()
        self.convert_into_distmatrices_contactmaps()
        self.convert_into_fasta()

    def apply(self):
        defected_pdbs = []
        our_pdbs = []
        pdb_lens = []
        i = 0
        for pdb_code in self.pdb_identifiers:
            # download
            self.pdbl.retrieve_pdb_file(
                pdb_code, pdir=CONSTANTS.PDB_DIR, file_format=CONSTANTS.CIF)
            # reading pdb file
            pdb_filename = CONSTANTS.PDB_DIR + pdb_code + CONSTANTS.CIF_EXT
            structure = self.parser.get_structure(pdb_code, pdb_filename)
            # getting all residues
            all_residues = structure.get_residues()
            # only amino acid residues
            aa_residues, seq, _ = self.filter_aa_residues(all_residues)
            n_aa_residues = len(aa_residues)
            # applying length filter
            if n_aa_residues >= self.min_len and n_aa_residues <= self.max_len:
                dist_matrix = np.zeros(
                    (n_aa_residues, n_aa_residues), np.float)
                try:
                    # compute distance matrix
                    dist_matrix = self.compute_distance_matrix(
                        aa_residues, aa_residues)
                    print(i, ",", pdb_code, "=====: ", n_aa_residues)
                    pdb_lens.append(n_aa_residues)
                    our_pdbs.append(pdb_code)
                    i += 1
                except Exception as e:
                    defected_pdbs.append(pdb_code)
                    continue
                # compute comtact map based on threshhold
                contact_map = np.where(dist_matrix < self.threshhold, 1, 0)
                filename = pdb_code + CONSTANTS.CSV_EXT
                # save contact_map and dist_matrix
                Utility.save_distance_matrix(dist_matrix, pdb_code)
                Utility.save_contact_map(contact_map, pdb_code)
                Utility.save_fasta_seq(seq, pdb_code)
                # show every contact_map and dist_matrix
                if i % self.view_every == 0:
                    self.view(filename)  # draws dist_matrix, contact_map
                # when we will get keep_n_pdbs, break the loop
                if i == self.keep_n_pdbs:
                    break
        # save our_pdbs, and defected_pdbs in file
        Utility.save_itemlist(defected_pdbs, CONSTANTS.DEFECTED_PDB_IDS)
        Utility.save_itemlist(our_pdbs, CONSTANTS.N_PDB_IDS)
        print(pdb_lens)

    def download(self):
        """
        download pdb file from PDB website
        """
        for identifier in self.pdb_identifiers:
            self.pdb_code = identifier
            self.pdbl.retrieve_pdb_file(
                self.pdb_code, pdir=CONSTANTS.PDB_DIR, file_format=CONSTANTS.CIF)

    def convert_into_distmatrices_contactmaps(self):
        """
            convert pdb file to distance matrix and contact maps for some threshhold
            for amino acid alpha-carbon residues

        """
        defected_pdb_ids = []
        all_pdb_lens = []
        for identifier in self.pdb_identifiers:
            self.pdb_code = identifier
            pdb_filename = CONSTANTS.PDB_DIR + self.pdb_code + CONSTANTS.CIF_EXT
            structure = self.parser.get_structure(
                self.pdb_code, pdb_filename)
            all_residues = structure.get_residues()
            aa_residues, non_aa_residues = self.filter_aa_residues(
                all_residues)
            dist_matrix = np.zeros(
                (len(aa_residues), len(aa_residues)), np.float)
            try:
                dist_matrix = self.compute_distance_matrix(
                    aa_residues, aa_residues)
                print(self.pdb_code, "============: ", len(aa_residues))
                all_pdb_lens.append(len(aa_residues))
            except Exception as e:
                defected_pdb_ids.append(self.pdb_code)
                continue
            contact_map = np.where(dist_matrix < self.threshhold, 1, 0)
            filename = self.pdb_code + CONSTANTS.CSV_EXT
            # Utility.save_distance_matrix(dist_matrix, self.pdb_code)
            Utility.save_contact_map(contact_map, self.pdb_code)
            # self.view(filename) # draws dist_matrix, contact_map
        Utility.save_itemlist(defected_pdb_ids, CONSTANTS.DEFECTED_PDB_IDS)
        print(all_pdb_lens)

    def convert_into_fasta(self):
        """
            convert each pdb file to fasta sequence format
        """
        for pdb_code in self.pdb_identifiers:
            from_cif = CONSTANTS.PDB_DIR + pdb_code + CONSTANTS.CIF_EXT
            to_fasta = CONSTANTS.FASTA_DIR + pdb_code + CONSTANTS.FASTA_EXT
            records = SeqIO.parse(from_cif, "cif-atom")
            count = SeqIO.write(records, to_fasta, "fasta")

    def compute_res_res_distance(self, residue_1, residue_2):
        """
        compute distance of two residue's alpha-carbon's coordinates
        """
        diff_vector = residue_1["CA"].coord - residue_2["CA"].coord
        return np.sqrt(np.sum(diff_vector * diff_vector))

    def compute_distance_matrix(self, chain_1, chain_2):
        """
        compute distance matrix of two chains
        """
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
        seq = ""
        for i in chain:
            if i.get_resname() in standard_aa_names:
                aa_residues.append(i)
                seq += self.aa_3to1[i.get_resname()]
            else:
                non_aa.append(i.get_resname())
                non_aa_residues.append(i.get_resname())
        # print(seq)
        # print(len(seq), len(aa_residues))
        return aa_residues, seq, non_aa_residues

    def read(self, filename, path):
        return np.loadtxt(path + filename, delimiter=',')

    def get_pdb_identifiers(self, file):
        file_content = open(file).read()
        file_content = file_content.lower()
        return file_content.split()

    def view(self, filename):
        """
            draw distance matrix and contact maps
        """
        img1 = self.read(filename, CONSTANTS.DISTANCE_MATRIX_DIR)
        img2 = self.read(filename, CONSTANTS.CONTACT_MAP_DIR)
        images = [img1, img2]
        titles = ["distance matrix", "contact map"]
        Utility.plot_images(images, filename, titles, cols=2)
        # self.__plot_images(images, filename, titles, cols=2)

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
