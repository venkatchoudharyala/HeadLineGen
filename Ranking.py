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
