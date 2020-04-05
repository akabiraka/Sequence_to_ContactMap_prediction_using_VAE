import numpy as np
from Bio.PDB import *
import matplotlib.pyplot as plt
import cv2


class PDBConverter(object):
    """docstring for PDBConverter."""

    def __init__(self):
        super(PDBConverter, self).__init__()

        self.pdb_identifies_file = '../inputs/pdb_identifiers.txt'
        self.pdb_identifiers = self.get_pdb_identifiers(
            self.pdb_identifies_file)
        self.pdb_code = '2blq'
        self.file_format = 'mmCif'
        self.pdir = "../pdbs/"
        self.file_extension = 'cif'
        self.pdb_filename = self.pdir + self.pdb_code + '.' + self.file_extension
        self.distance_matrices_dir = "../distance_matrices/"
        self.contact_maps_dir = "../contact_maps/"
        self.threshhold = 12.0

        self.pdbl = PDBList()
        self.parser = MMCIFParser()

    def solve(self):
        for identifier in self.pdb_identifiers:
            self.pdb_code = identifier
            self.pdbl.retrieve_pdb_file(
                self.pdb_code, pdir=self.pdir, file_format=self.file_format)
            structure = self.parser.get_structure(
                self.pdb_code, self.pdb_filename)
            model = structure[0]
            dist_matrices, contact_maps = self.compute_dist_matrices(
                model, th=self.threshhold)
            for i in range(len(dist_matrices)):
                filename = self.pdb_code + "_" + str(i) + ".csv"
                self.save(dist_matrices[i], filename,
                          self.distance_matrices_dir)
                self.save(contact_maps[i], filename, self.contact_maps_dir)

                img = self.read(filename, self.contact_maps_dir)
                plt.imshow(img)
                plt.show()

    def compute_res_res_distance(self, residue_1, residue_2):
        diff_vector = residue_1["CA"].coord - residue_2["CA"].coord
        return np.sqrt(np.sum(diff_vector * diff_vector))

    def compute_chain_chain_distance_matrix(self, chain_1, chain_2):
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

    def compute_dist_matrices(self, model, th=12.0):
        """
        th: threshhold
            default: 12 angstroms
        """
        dist_matrices = []
        contact_maps = []
        for chain_1 in model:
            chain_1_aa_residues, _ = self.filter_aa_residues(chain_1)
            for chain_2 in model:
                chain_2_aa_residues, _ = self.filter_aa_residues(chain_2)
                dist_matrix = self.compute_chain_chain_distance_matrix(
                    chain_1_aa_residues, chain_2_aa_residues)
                contact_map = np.where(dist_matrix < th, 1, 0)
                dist_matrices.append(dist_matrix)
                contact_maps.append(contact_map)

        return dist_matrices, contact_maps

    def save(self, matrix, filename, path):
        np.savetxt(path + filename, matrix, delimiter=',')

    def read(self, filename, path):
        return np.loadtxt(path + filename, delimiter=',')

    def get_pdb_identifiers(self, file):
        file_content = open(file).read()
        return file_content.split()
