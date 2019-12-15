import csv
from typing import List


def list_to_csv(list: List[str], output_path: str, header: str = ""):
    with open(output_path, 'w', newline='') as writeFile:
        writer = csv.writer(writeFile,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header != "": writer.writerow([header])
        for element in list:
            writer.writerow([element])
    writeFile.close()
