# Resume-NER

# About

This repository applies BERT for named entity recognition on resumes. The goal is to find useful information present in resume.

# Requirements

```bash
pip3 install -r requirements.txt
```

# Training

To train model use:
```bash
python3 train.py
``` 
optional arguments:

-e epochs &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of epochs

-o path &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; output path to save model state

# Get output with Normalizer

cd root dir Resume-NER-master
put resume(s) into "input" file
run python batch_infer.py
get result(s) in "output" file


```
Links:&nbsp;&nbsp;
[Actual Resume](NER/demo/Resume%20-%20Ayush%20Srivastava.pdf)
&nbsp;&nbsp;&nbsp;&nbsp;
[Full Response](NER/demo/response.json)

# Links
[Dataset](https://www.kaggle.com/dataturks/resume-entities-for-ner)
