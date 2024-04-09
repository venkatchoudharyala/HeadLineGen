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
        if article_info[1]==1:
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
