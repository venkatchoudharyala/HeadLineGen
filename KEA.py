# Key Phrase Extraction

import textacy
from textacy import *
import string

def remove_punctuation(text):
	table = str.maketrans('', '', string.punctuation)
	return text.translate(table)

def KeyPhraseSGRank(Article):
	en = textacy.load_spacy_lang("en_core_web_sm")
	
	Article = remove_punctuation(Article)
	
	doc = textacy.make_spacy_doc(Article, lang=en)
	
	TopPhrases = [kps for kps, weights in textacy.extract.keyterms.sgrank(doc, ngrams = (1, 3), topn=1.0)]
	print(TopPhrases)
	return TopPhrases
