from Bio import SeqIO
import os
from pdb_converter import PDBConverter

pdb_converter = PDBConverter()
pdb_converter.do()


# records = SeqIO.parse("../pdbs/2blq.cif", "cif-atom")
# count = SeqIO.write(records, "../fastas/2blq.fasta", "fasta")
# print("Converted %i records" % count)
#
#
# records = SeqIO.parse("../fastas/2ok6.fasta", "fasta")
# for record in records:
#     print("Id: %s" % record.id)
#     print("Name: %s" % record.name)
#     print("Description: %s" % record.description)
#     print("Annotations: %s" % record.annotations)
#     print("Sequence Data: %s" % record.seq)
#     print("Sequence Alphabet: %s" % record.seq.alphabet)
