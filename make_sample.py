import pandas as pd

df = pd.read_csv("data/sps-corpus-1.0-2025-11-25-en/ss-corpus-en.tsv", sep="\t")

sample = df.sample(n=16, random_state=42)

sample.to_csv("./data/test_sample/sample_md.tsv", sep="\t", index=False)