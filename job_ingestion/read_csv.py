import pandas as pd

df = pd.read_csv("data/it_job_postings_annotated.csv")

print(df[['skills_annotated_raw','skills_annotated']].head(20))