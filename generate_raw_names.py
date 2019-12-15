import os
import csv
import faker
from typing import List

faker.Faker.seed(4321)
approx_name_count = 20
output_path = 'test/data/individuals_generated.csv'
output_csv_header = "name"
faker_regions = [
"en_AU",
"en_CA",
"en_GB",
"en_NZ",
"en_US"
]


def generate_individual_names(regions: List[str], approx_name_count: int) -> List[str]:
    generated_full_names = []
    iterations_per_region = int(approx_name_count / len(regions))
    for region in regions:
        factory = faker.Faker(region)
        generated_full_names = generated_full_names + [factory.name() for i in range(0, iterations_per_region)]

    distinct_full_names = [*{*generated_full_names}]
    return distinct_full_names


def list_to_csv(list: List[str], header: str):
    with open(output_path, 'w', newline='') as writeFile:
        writer = csv.writer(writeFile,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([header])
        for element in list:
            writer.writerow([element])
    writeFile.close()

if __name__ == "__main__":

    distinct_full_names = generate_individual_names(faker_regions, approx_name_count)
    list_to_csv(distinct_full_names, output_csv_header)


