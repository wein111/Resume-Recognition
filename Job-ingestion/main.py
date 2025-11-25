import pandas as pd
import re, json
from helpers import clean_csv, normalize_company, process_job_titles, process_locations, pattern_matching

df = clean_csv()
df = normalize_company(df)
df = process_job_titles(df)
df = process_locations(df)
df = pattern_matching(df)
print(df.columns.to_list())

# ['jobpost', 'date', 'Title', 'Company', 'AnnouncementCode', 'Term',
#       'Eligibility', 'Audience', 'StartDate', 'Duration', 'Location',
#       'JobDescription', 'JobRequirment', 'RequiredQual', 'Salary',
#       'ApplicationP', 'OpeningDate', 'Deadline', 'Notes', 'AboutC', 'Attach',
#       'Year', 'Month', 'IT', 'company_norm', 'title_clean', 'location_clean', 'skills_text', 'skills_norm']

# for i, row in has_skills.head(100).iterrows():
#     print(f"({row['Company']})({row['Title']}): {row['skills_norm']}")

out = df[["company_norm", "title_clean", "location_clean", "skills_norm"]]
out.to_json("output/job_postings_extracted.jsonl", orient="records", lines=True)