import numpy as np
import random

import constants as CONSTANTS
import utility as Utility


class DataSpliter(object):
    """docstring for DataSpliter."""

    def __init__(self, file=CONSTANTS.ALL_PDB_IDS):
        super(DataSpliter).__init__()
        pdb_identifies_file = file
        self.pdb_identifiers = Utility.get_pdb_identifiers(pdb_identifies_file)

    def split(self, train_size=.60, val_size=.20):
        # test_size = .20
        random.shuffle(self.pdb_identifiers)
        to_train_index = round(len(self.pdb_identifiers) * train_size)
        train_set = self.pdb_identifiers[0:to_train_index]
        remaining = self.pdb_identifiers[to_train_index:]
        to_val_index = round(len(self.pdb_identifiers) * val_size)
        val_set = remaining[0:to_val_index]
        test_set = remaining[to_val_index:]
        print("train size:", len(train_set), "val size:",
              len(val_set), "test size:", len(test_set))
        Utility.save_itemlist(train_set, CONSTANTS.TRAIN_FILE)
        Utility.save_itemlist(val_set, CONSTANTS.VAL_FILE)
        Utility.save_itemlist(test_set, CONSTANTS.TEST_FILE)
        return train_set, val_set, test_set
