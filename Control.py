import re
import os
from nltk.parse.corenlp import CoreNLPServer

import PostProcess as PP
import Ranking as rank
import Compression as comp
import KEA as kea
import LeadingSentences as ls

os.environ['CLASSPATH'] = 'stanford-corenlp-4.5.6'
server = CoreNLPServer()
server.start()

def Generate(Article):
  cleaned_article = re.sub(r'\([^)]*\)', '', Article)

  KeyPhrases = kea.KeyPhraseSGRank(cleaned_article)

  LSG = ls.LeadSentencesOOPS(cleaned_article)
  LeadingSentences  = LSG.leading_sentences()
  #LeadingSentences = leading_sentences(cleaned_article)
  #LeadingSentences = get_first_sentences(cleaned_article)
  print(LeadingSentences)

  CompressedSentences = comp.CompressionAlgorithm(LeadingSentences, KeyPhrases, server)

  ResultDict = rank.Ranking(CompressedSentences, KeyPhrases)

  max_key = max(ResultDict, key=lambda k: ResultDict[k])
  print(ResultDict)
  return pp.PostProcess(max_key)
