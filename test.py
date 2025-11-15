import torch
from transformers import BertTokenizerFast, BertForTokenClassification


MODEL_PATH = "model-state.bin"  #Output model path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  1+10 tags
tags_vals = [
    "O",
    "B-Name", "I-Name",
    "B-Degree", "I-Degree",
    "B-Skills", "I-Skills",
    "B-College Name", "I-College Name",
    "B-Email Address", "I-Email Address",
    "B-Designation", "I-Designation",
    "B-Companies worked at", "I-Companies worked at",
    "B-Graduation Year", "I-Graduation Year",
    "B-Years of Experience", "I-Years of Experience",
    "B-Location", "I-Location"
]

tag2idx = {tag: i for i, tag in enumerate(tags_vals)}
idx2tag = {i: tag for i, tag in enumerate(tags_vals)}

MAX_LEN = 128

# --------------------------
# initialize tokenizer and model
# --------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict']
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()



def ner_predict(text):
    # Tokenize
    encoding = tokenizer.encode_plus(
        text,
        return_tensors="pt",
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    offsets = encoding["offset_mapping"][0]

    # prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs[0]  # shape: (1, seq_len, num_labels)

    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    #revert  tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # print result
    result = []
    for idx, pred_id in enumerate(predictions):
        if tokens[idx] in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        tag = idx2tag[pred_id]
        token = tokens[idx]
        # combine WordPiece
        if token.startswith("##") and result:
            result[-1][0] += token[2:]
        else:
            result.append([token, tag])
    return result



#text = "John Doe graduated from MIT in 2020 and works at Google as a Software Engineer."
#text = "Abhishek Jha\nApplication Development Associate - Accenture\n\nBengaluru, Karnataka - Email me on Indeed: indeed.com/r/Abhishek-Jha/10e7a8cb732bc43a\n\n• To work for an organization which provides me the opportunity to improve my skills\nand knowledge for my individual and company's growth in best possible ways.\n\nWilling to relocate to: Bangalore, Karnataka\n\nWORK EXPERIENCE\n\nApplication Development Associate\n\nAccenture -\n\nNovember 2017 to Present\n\nRole: Currently working on Chat-bot. Developing Backend Oracle PeopleSoft Queries\nfor the Bot which will be triggered based on given input. Also, Training the bot for different possible\nutterances (Both positive and negative), which will be given as\ninput by the user.\n\nEDUCATION\n\nB.E in Information science and engineering\n\nB.v.b college of engineering and technology -  Hubli, Karnataka\n\nAugust 2013 to June 2017\n\n12th in Mathematics\n\nWoodbine modern school\n\nApril 2011 to March 2013\n\n10th\n\nKendriya Vidyalaya\n\nApril 2001 to March 2011\n\nSKILLS\n\nC (Less than 1 year), Database (Less than 1 year), Database Management (Less than 1 year),\nDatabase Management System (Less than 1 year), Java (Less than 1 year)\n\nADDITIONAL INFORMATION\n\nTechnical Skills\n\nhttps://www.indeed.com/r/Abhishek-Jha/10e7a8cb732bc43a?isid=rex-download&ikw=download-top&co=IN\n\n\n• Programming language: C, C++, Java\n• Oracle PeopleSoft\n• Internet Of Things\n• Machine Learning\n• Database Management System\n• Computer Networks\n• Operating System worked on: Linux, Windows, Mac\n\nNon - Technical Skills\n\n• Honest and Hard-Working\n• Tolerant and Flexible to Different Situations\n• Polite and Calm\n• Team-Player"
text = "Sarah Chen has worked at Microsoft and Amazon as a Product Manager for 3 years, using python, SQL, word and powerpoint."
text2 = "John Doe graduated from Stanford University and works at Google"
prediction = ner_predict(text2)

for token, tag in prediction:
    print(f"{token}\t{tag}")
