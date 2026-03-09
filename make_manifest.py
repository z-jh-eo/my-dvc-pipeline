import pandas as pd


def make_manifest(file_path):
    md = pd.read_csv(file_path, sep="\t")
