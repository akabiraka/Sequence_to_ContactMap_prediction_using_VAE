from Bio.PDB import *
from Bio import SeqIO

import constants as CONSTANTS
import utility as Utility


class Filters(object):
    """docstring for Filters."""

    def __init__(self, pdb_file):
        super(Filters, self).__init__()
        self.pdb_identifies_file = pdb_file  # CONSTANTS.ALL_PDB_IDS
        self.output_file = CONSTANTS.PDB_IDS_2000
        self.pdb_identifiers = Utility.get_pdb_identifiers(
            self.pdb_identifies_file)
        self.parser = MMCIFParser()

    def apply_length(self, min, max):
        valid_seqs = []
        i = 1
        for pdb_code in self.pdb_identifiers:
            pdb_filename = CONSTANTS.PDB_DIR + pdb_code + CONSTANTS.CIF_EXT
            structure = self.parser.get_structure(pdb_code, pdb_filename)
            all_residues = structure.get_residues()
            aa_residues, non_aa_residues = self.filter_aa_residues(
                all_residues)
            if len(aa_residues) >= min and len(aa_residues) <= max:
                print(i, ":", pdb_code)
                i += 1
                valid_seqs.append(pdb_code)
            if i == 2000:
                break
        print(len(valid_seqs))
        Utility.save_itemlist(valid_seqs, self.output_file)

    def filter_aa_residues(self, all_residues):
        """
        a chain can be heteroatoms(water, ions, etc; anything that isn't an amino acid or nucleic acid)
        so this function get rid of atoms excepts amino-acids
        """
        aa_residues = []
        non_aa_residues = []
        non_aa = []
        for i in all_residues:
            if i.get_resname() in standard_aa_names:
                aa_residues.append(i)
            else:
                non_aa.append(i.get_resname())
                non_aa_residues.append(i)
        return aa_residues, non_aa_residues
