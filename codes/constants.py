
WINDOW_SIZE = 32
WINDOW_STRIDE = 8

# directories
PDB_DIR = "../pdbs/"
DISTANCE_MATRIX_DIR = "../distance_matrices/"
CONTACT_MAP_DIR = "../contact_maps/"
FASTA_DIR = "../fastas/"
FEATURE_DIR = "../features/"

# file format
CIF = 'mmCif'
FASTA = 'fasta'

# extensions
CIF_EXT = ".cif"
CSV_EXT = ".csv"
FASTA_EXT = ".fasta"
PT_EXT = ".pt"

# amino acid letters from wiki
AMINO_ACID = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
              'T', 'U', 'V', 'W', 'Y', 'X', 'Z']
AMINO_ACID_20 = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
                 'W', 'Y']
AMINO_ACID_3TO1 = {'ALA': 'A',
                   'CYS': 'C',
                   'ASP': 'D',
                   'GLU': 'E',
                   'PHE': 'F',
                   'GLY': 'G',
                   'HIS': 'H',
                   'ILE': 'I',
                   'LYS': 'K',
                   'LEU': 'L',
                   'MET': 'M',
                   'ASN': 'N',
                   'PRO': 'P',
                   'GLN': 'Q',
                   'ARG': 'R',
                   'SER': 'S',
                   'THR': 'T',
                   'VAL': 'V',
                   'TRP': 'W',
                   'TYR': 'Y'}

# feature padding name
ONE_HOT_ = "_one_hot"

# input files
# ALL_PDB_IDS = '../inputs/dncon_pdb_identifiers.txt'
# ALL_PDB_IDS = '../inputs/pdb_identifiers.txt'
ALL_PDB_IDS = "../inputs_1/pdb_id_list.txt"
N_PDB_IDS = '../inputs/n_pdb_ids.txt'
DEFECTED_PDB_IDS = '../inputs/defected_pdb_ids.txt'
TRAIN_FILE = '../inputs/train.txt'
VAL_FILE = '../inputs/val.txt'
TEST_FILE = '../inputs/test.txt'
# TEST_FILE = '../inputs/test_tiny.txt'

LOG_FILE = "../outputs/running_log.txt"
