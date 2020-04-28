import os
import constants as CONSTANTS

from pdb_converter import PDBConverter
from data_spliter import DataSpliter
from feature_generator import FeatureGenerator

pdb_converter = PDBConverter(CONSTANTS.ALL_PDB_IDS)
pdb_converter.apply()


feature_generator = FeatureGenerator(CONSTANTS.N_PDB_IDS)
feature_generator.one_hot()

data_spliter = DataSpliter(CONSTANTS.N_PDB_IDS)
data_spliter.split()
