# Parsing and Compression Algo

import string
from nltk.parse.corenlp import CoreNLPParser
from nltk.tree.tree import Tree

def remove_punctuation(text):
  table = str.maketrans('', '', string.punctuation)
  return text.translate(table)

def Parsing(Sentence, server):
  parser = CoreNLPParser(url=server.url)
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
