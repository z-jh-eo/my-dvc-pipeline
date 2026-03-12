import pandas as pd

df = pd.read_csv("./data/sps-corpus-1.0-2025-11-25-en/ss-corpus-en.tsv", sep="\t")

sample = df.sample(n=16, random_state=42)

sample.to_csv("./data/test_sample/sample_md.tsv", sep="\t", index=False)

df = pd.read_csv("./data/sps-corpus-2.0-2025-12-05-fr/ss-corpus-fr.tsv", sep="\t")

sample = df.sample(n=16, random_state=42)

sample.to_csv("./data/test_sample/sample_md_fr.tsv", sep="\t", index=False)

