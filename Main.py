import requests
import zipfile
import os
import nltk

def Downloader():
  directory_path = "Parser/stanford-corenlp-4.5.6"
	
	if os.path.exists(directory_path) and os.listdir(directory_path):
    pass
  else:
    nltk.download('punkt')
    nltk.download('stopwords')
    url = "https://nlp.stanford.edu/software/stanford-corenlp-4.5.6.zip"
    
    filename = "stanford-corenlp-4.5.6.zip"
    
    directory = "./Parser/"
    
    os.makedirs(directory, exist_ok=True)
    
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(os.path.join(directory, filename), 'wb') as f:
            f.write(response.content)
        print("Download successful.")
    
        with zipfile.ZipFile(os.path.join(directory, filename), 'r') as zip_ref:
            zip_ref.extractall(directory)
        print("Extraction successful.")
    else:
        print("Failed to download file.")


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

import re
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy.spatial import distance
import networkx as nx

class LeadSentencesOOPS:
    def __init__(self, df):
        self.df = df
        self.sentences = sent_tokenize(self.df)

    def pre_process(self):
        sentences_clean = [re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in self.sentences]
        stop_words = stopwords.words('english')
        sentence_tokens = [[words for words in sentence.split(' ') if words not in stop_words] for sentence in sentences_clean]
        return sentence_tokens

    # def count_paragraphs(self):
    #     val=self.df
    #     paragraphs = re.split(r"\n\n+",val)
    #     num_paragraphs = len(paragraphs)
    #     print("num_paragraphs: ",num_paragraphs)

    def count_paragraphs(self):
        text=self.df
        paragraphs = re.split(r'\n\s*\n', text)
        return (paragraphs,len(paragraphs))

    def word2vec(self):
        sentence_tokens = self.pre_process()
        w2v = Word2Vec(sentence_tokens, vector_size=1, min_count=1, epochs=1500)
        sentence_embeddings = []
        max_len = max(len(tokens) for tokens in sentence_tokens)
        for words in sentence_tokens:
            embedding = [w2v.wv[word] for word in words]
            padding_length = max_len - len(embedding)
            padded_embedding = np.pad(embedding, [(0, padding_length), (0, 0)], mode='constant')
            sentence_embeddings.append(padded_embedding)
        return sentence_embeddings

    def similarity_matrix(self):
        sentence_tokens = self.pre_process()
        sentence_embeddings = self.word2vec()
        similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])
        for i, row_embedding in enumerate(sentence_embeddings):
            for j, column_embedding in enumerate(sentence_embeddings):
                similarity_matrix[i][j] = 1 - distance.cosine(row_embedding.ravel(), column_embedding.ravel())
        return similarity_matrix

    def num_of_leadingsentences(self):
        num_sentences = len(self.sentences)
        if num_sentences < 5:
            top = 1
        elif num_sentences < 10:
            top = 2
        elif num_sentences < 25:
            top = 4
        elif num_sentences < 50:
            top = 9
        elif num_sentences < 100:
            top = 18
        elif num_sentences < 200:
            top = 25
        elif num_sentences >= 201:
            top = 40
        return top

    def text_rank(self,num_sentences_to_extract):
        li=[]
        similarity_matrixs = self.similarity_matrix()
        nx_graph = nx.from_numpy_array(similarity_matrixs)
        scores = nx.pagerank(nx_graph)
        top_sentence = {sentence: scores[index] for index, sentence in enumerate(self.sentences)}
        top = dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:num_sentences_to_extract])
        for sent in self.sentences:
            if sent in top.keys():
                li.append(sent)
        return li

    def leading_sentences(self):
        article_info = self.count_paragraphs()
        leading_sentences=[]
        #if there is only one para in article then num_of_leading sentences are selected based on fixed constant
        if article_info[1] <= 3:
          num_sentences_to_extract=self.num_of_leadingsentences()
          LSG_article = LeadSentencesOOPS(str(article_info[0]))
          leading_sentences.extend(LSG_article.text_rank(num_sentences_to_extract))
          #leading_sentences_corpus.append(leading_sentences)
        else:
          num_sentences_to_extract=1                   #if there are more than one paras in article
          paragraphs = article_info[0]
          #print("num_paras: ",paragraphs)
          #extracting one leading sentence from each paragraph
          for para in paragraphs:
              LSG = LeadSentencesOOPS(para)
              output = LSG.text_rank(num_sentences_to_extract)
              leading_sentences.extend(output)
          #extractig leading sentence from entire article
          LSG_article = LeadSentencesOOPS(para)
          leading_sentences.extend(LSG_article.text_rank(num_sentences_to_extract))

        return leading_sentences

# Parsing and Compression Algo

import string
from nltk.parse.corenlp import CoreNLPParser
from nltk.tree.tree import Tree

def remove_punctuation(text):
  table = str.maketrans('', '', string.punctuation)
  return text.translate(table)

def Parsing(Sentence, url):
  parser = CoreNLPParser(url=url)
  return next(parser.raw_parse(Sentence))

def find_leftmost_S(tree):
    if isinstance(tree, str):  # Terminal node
        return None
    elif tree.label() == 'S':  # Found leftmost S node
        return tree
    else:
        for subtree in tree:
            result = find_leftmost_S(subtree)
            if result is not None:
                return result

def Pruning(tree, Label):
  if isinstance(tree, str):
    return tree
  if tree.height() > 0:
    filtered_children = [Pruning(child, Label) for child in tree if (isinstance(child, str) or child.height() > 0) and (isinstance(child, str) or child.label() != Label)]
    return Tree(tree.label(), filtered_children)
  else:
    return tree

def IterativeTrimming(HeadLine, SGRankList, Threshold):
  if len(HeadLine) > Threshold:
    if len(SGRankList) > 0:
      ptr = SGRankList[-1]
    else:
      return HeadLine
    if HeadLine.find(ptr) > 0:
      if HeadLine[HeadLine.find(ptr) - 1] != ' ':
        HeadLine = HeadLine.replace(ptr, ":", 1)
      else:
        HeadLine = HeadLine.replace(' ' + ptr, "", 1)
    else:
      HeadLine = HeadLine.replace(ptr + ' ', "", 1)
    return IterativeTrimming(HeadLine, SGRankList[: len(SGRankList) - 1], Threshold)
  else:
    return HeadLine

def Extract(Treex):
    k = Treex.leaves()
    Trex = ''
    for i in k:
      Trex += i + ' '
    return Trex

def CompressionAlgorithm(LeadSents, TopPhrases, server):
  CompressedSentences = []
  for i in LeadSents:
    Suppy = remove_punctuation(i)

    ParsedSentence = Parsing(Suppy, server)

    for i in ParsedSentence:
      for j in i:
        lefts = find_leftmost_S(j)
        if lefts is not None:
          LeftMostS = lefts
        else:
          LeftMostS = i
        break

    Labels = [ 'SBAR', 'DT', 'TMP', 'CC']
    for i in Labels:
      Temp = Pruning(LeftMostS, i)
      LeftMostS = Temp

    Trex = Extract(Temp)
    Kalix = IterativeTrimming(Trex, TopPhrases, 120)

    '''PS = Parsing(Kalix, server)
    Tk = Pruning(PS, 'SBAR')

    Trex = Extract(Tk)'''

    CompressedSentences.append(Kalix)
  return CompressedSentences

# Key Phrase Matching and Ranking

def SGRMatching(HeadLine, TopPhrases):
  l, Flag, itre = len(TopPhrases), 0.0, 0
  for Phrase in TopPhrases:
    if Phrase in HeadLine:
      Flag += (l - TopPhrases.index(Phrase)) / l
      itre += 1
  return (itre * Flag) / l
  '''
  if itre != 0:
    return Flag / itre
  else:
    return -1'''

def Ranking(CompressedSentences, KeyPhrases):
  ResultDict = {}
  for i in CompressedSentences:
    ResultDict[i] = SGRMatching(i, KeyPhrases)
  return ResultDict

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

import re
import os
from nltk.parse.corenlp import CoreNLPServer

os.environ['CLASSPATH'] = 'stanford-corenlp-4.5.6'
server = CoreNLPServer()
server.start()

def Generate(Article):
  Dowloader()
  os.environ['CLASSPATH'] = 'Parser/stanford-corenlp-4.5.6'
  
  response = requests.get('http://localhost:9000')
  if response.status_code == 200:
    print("CoreNLP server is up and running!")
    url = 'http://localhost:9000'
  else:
    print("CoreNLP server is not running.")
    server = CoreNLPServer()
    server.start()
    url = server.url
  
  cleaned_article = re.sub(r'\([^)]*\)', '', Article)

  KeyPhrases = KeyPhraseSGRank(cleaned_article)

  LSG = LeadSentencesOOPS(cleaned_article)
  LeadingSentences  = LSG.leading_sentences()
  #LeadingSentences = leading_sentences(cleaned_article)
  #LeadingSentences = get_first_sentences(cleaned_article)
  print(LeadingSentences)

  CompressedSentences = CompressionAlgorithm(LeadingSentences, KeyPhrases, url)

  ResultDict = Ranking(CompressedSentences, KeyPhrases)

  max_key = max(ResultDict, key=lambda k: ResultDict[k])
  print(ResultDict)
  return PostProcess(max_key)
