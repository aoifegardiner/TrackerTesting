# src/utils.py

import os
import yaml
import dload
from pathlib import Path

# Define project root as /Workspace/agardiner_STIR_submission
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def resolve_path(*parts):
    """Helper: resolve paths relative to project root."""
    return PROJECT_ROOT.joinpath(*parts)


class CaseSample:
    def __init__(self, case_id, case_sample_path, case_sample_links, anchors):
        self.case_id = case_id
        self.case_sample_path = Path(case_sample_path)
        self.case_sample_links = case_sample_links
        self.anchors = anchors

    def download_case_sample_data(self):
        print(f"\t{self.case_sample_path}")
        make_dir_if_needed(self.case_sample_path)
        for file_name, file_link in self.case_sample_links.items():
            file_path = self.case_sample_path / file_name
            if file_path.is_file():
                print(f"\t\t{file_path}: CHECK")
            else:
                print(f"\t\t{file_path}: DOWNLOADING...")
                dload.save(file_link, str(file_path))


class Case:
    def __init__(self, case_id):
        self.case_id = case_id
        self.case_samples = []


def is_path_file(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(f"File not found: {string}")


def load_yaml_data(path):
    path = resolve_path(path)  # Always resolve relative to project root
    if is_path_file(path):
        with open(path) as f_tmp:
            return yaml.load(f_tmp, Loader=yaml.FullLoader)


def write_yaml_data(path, data):
    path = resolve_path(path)
    with open(path, "w") as fp:
        yaml.dump(data, fp)


def make_dir_if_needed(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_cases(config_data):
    dataset_dir = resolve_path(config_data["dir"], config_data["subdir"])
    anchor_data = config_data["anchors"]
    cases = []

    # Loop through cases
    for case_id, case_data in config_data["cases"].items():
        case = Case(case_id)
        path_case = dataset_dir / case_id

        # Loop through samples of each case
        for sample_number, sample_links in case_data.items():
            anchors = anchor_data[case_id][sample_number]
            path_case_sample = path_case / sample_number
            cs = CaseSample(case_id, path_case_sample, sample_links, anchors)
            case.case_samples.append(cs)

        cases.append(case)

    return cases


def download_folder(config_data):
    if config_data.get("is_to_download", False):
        print(f"DOWNLOAD: Checking `{config_data['subdir']}` data:")
        cases = get_cases(config_data)
        for case in cases:
            for cs in case.case_samples:
                cs.download_case_sample_data()


def download_data(config):
    download_folder(config["validation"])
    download_folder(config["test"])
