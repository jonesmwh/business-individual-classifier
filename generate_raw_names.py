import faker
from typing import List
from utils import list_to_csv, load_config

config = load_config()

faker.Faker.seed(4321)
name_count = config["generate_raw_names"]["name_count"].get(int)
output_path_individual = config["rel_paths"]["raw_data_root"].get(str) + config["generate_raw_names"]["individual_filename"].get(str)
output_path_business = config["rel_paths"]["raw_data_root"].get(str) + config["generate_raw_names"]["business_filename"].get(str)
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

def generate_business_names(regions: List[str], approx_name_count: int) -> List[str]:
    generated_full_names = []
    iterations_per_region = int(approx_name_count / len(regions))
    for region in regions:
        factory = faker.Faker(region)
        generated_full_names = generated_full_names + [factory.company() for i in range(0, iterations_per_region)]

    distinct_full_names = [*{*generated_full_names}]
    return distinct_full_names



def run_name_generation():
    distinct_names_individual = generate_individual_names(faker_regions, name_count)
    list_to_csv(distinct_names_individual, output_path_individual, output_csv_header)

    distinct_names_business = generate_business_names(faker_regions, name_count)
    list_to_csv(distinct_names_business, output_path_business, output_csv_header)


if __name__ == "__main__":
    run_name_generation()
