# Post Processing using DistilBert

import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

#
# Split text to segments of length 200, with overlap 50
#
def split_to_segments(wrds, length, overlap):
    resp = []
    i = 0
    while True:
        wrds_split = wrds[(length * i):((length * (i + 1)) + overlap)]
        if not wrds_split:
            break

        resp_obj = {
            "text": wrds_split,
            "start_idx": length * i,
            "end_idx": (length * (i + 1)) + overlap,
        }

        resp.append(resp_obj)
        i += 1
    return resp


#
# Punctuate wordpieces
#
def punctuate_wordpiece(wordpiece, label):
    if label.startswith('UPPER'):
        wordpiece = wordpiece.upper()
    elif label.startswith('Upper'):
        wordpiece = wordpiece[0].upper() + wordpiece[1:]
    if label[-1] != '_' and label[-1] != wordpiece[-1]:
        wordpiece += label[-1]
    return wordpiece


#
# Punctuate text segments (200 words)
#
def punctuate_segment(wordpieces, word_ids, labels, start_word):
    result = ''
    for idx in range(0, len(wordpieces)):
        if word_ids[idx] == None:
            continue
        if word_ids[idx] < start_word:
            continue
        wordpiece = punctuate_wordpiece(wordpieces[idx][2:] if wordpieces[idx].startswith('##') else wordpieces[idx],
                            labels[idx])
        if idx > 0 and len(result) > 0 and word_ids[idx] != word_ids[idx - 1] and result[-1] != '-':
            result += ' '
        result += wordpiece
    return result


#
# Tokenize, predict, punctuate text segments (200 words)
#
def process_segment(words, tokenizer, model, start_word, encoder_max_length):

    tokens = tokenizer(words['text'],
                       padding="max_length",
                       # truncation=True,
                       max_length=encoder_max_length,
                       is_split_into_words=True, return_tensors='pt')

    with torch.no_grad():
        logits = model(**tokens).logits
    logits = logits.cpu()
    predictions = np.argmax(logits, axis=-1)

    wordpieces = tokens.tokens()
    word_ids = tokens.word_ids()
    id2label = model.config.id2label
    labels = [[id2label[p.item()] for p in prediction] for prediction in predictions][0]

    return punctuate_segment(wordpieces, word_ids, labels, start_word)


#
# Punctuate text of any length
#
def punctuate(text, tokenizer, model, encoder_max_length):
    text = text.lower()
    text = text.replace('\n', ' ')
    words = text.split(' ')

    overlap = 50
    slices = split_to_segments(words, 150, 50)

    result = ""
    start_word = 0
    for text in slices:
        corrected = process_segment(text, tokenizer, model, start_word, encoder_max_length)
        result += corrected + ' '
        start_word = overlap
    return result

def PostProcess(Sentence):
  checkpoint = "unikei/distilbert-base-re-punctuate"
  tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint)
  model = DistilBertForTokenClassification.from_pretrained(checkpoint)
  encoder_max_length = 256
  return punctuate(Sentence, tokenizer, model, encoder_max_length)
