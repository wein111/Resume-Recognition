Open cmd and go to the root directory, run "pip install -r requirements.txt"

The main normalizer is stored in the skill_normalizer file. The vocab.json and alias.json files are extracted from the lexicon provided by ESCO(data\skills_en.csv). These two files are extracted from the data\skills_en.csv file by the esco_to_normalizer.py script. 
To update these two files, modify and run the esco_to_normalizer.py file.
Check ESCO datasorce in reference.txt. 

The cli_demo.py file is used to test the normalizer, taking data\mock_skills.json as input.

Running "python run_all.py" in the root directory will call the normalizer, taking data\entities.json as input and generating the normalized_detailed.json and normalized_for_task4.json files in the data directory.