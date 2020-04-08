from Bio import SeqIO
import os
from pdb_converter import PDBConverter
from feature_generator import FeatureGenerator

# pdb_converter = PDBConverter()
# pdb_converter.do()

feature_generator = FeatureGenerator()
feature_generator.one_hot()
