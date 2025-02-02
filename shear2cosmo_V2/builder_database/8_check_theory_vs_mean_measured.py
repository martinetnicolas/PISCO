import re
from collections import defaultdict
from pathlib import Path

import numpy
from tqdm import tqdm

# --------------------------------------------------------------------------------------------------


FOLDER_BASE = Path("/net/GECO/pisco/SHEAR2COSMO")
DBB = "DBB_OPT_26"
NUMBER_OF_COSMO = 26
FOLDER_MEASURED = FOLDER_BASE / DBB / "measured"


# --------------------------------------------------------------------------------------------------


all_paths = sorted(list(FOLDER_MEASURED.glob("id*.txt")))
files_by_id = defaultdict(list)
pattern = re.compile(r"id(\d+)_seed0")
for path in all_paths:
    match = pattern.search(path.name)
    if match:
        file_id = int(match.group(1))
        files_by_id[file_id].append(path)
lists_by_id = [files_by_id[i] for i in range(NUMBER_OF_COSMO)]

for cosmo in tqdm(numpy.arange(NUMBER_OF_COSMO), total=NUMBER_OF_COSMO):
    number = len(lists_by_id[cosmo])
    K = numpy.zeros(10)
    for nb in range(number):
        K += numpy.loadtxt(lists_by_id[cosmo][nb])[:, 3]
    K /= number
    print(K)
