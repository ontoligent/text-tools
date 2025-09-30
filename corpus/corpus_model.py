import numpy as np
import pandas as pd 

class CorpusModel():
    
    def __init__(self, CORPUS:pd.DataFrame, doc_content_col='doc_content'):
        self.CORPUS = CORPUS
        self.n_docs = len(CORPUS)
        self.doc_content_col = doc_content_col
        # self.make_tables()  

    def make_tables(self):

        # Convert Corpus to Tokens
        self.TOKEN = self.CORPUS[self.doc_content_col].str.split(expand=True).stack().to_frame('token_str')
        self.TOKEN.index.names = ['doc_id', 'token_ord'] 
        self.TOKEN['term_str'] = self.TOKEN.token_str.str.lower().replace(r"\W+", "", regex=True)

        # Covert Tokens to Bag-of-Words
        self.BOW = self.TOKEN.groupby(['doc_id', 'term_str']).term_str.count().to_frame('tf')
        
        # Extract Vocabulary from Bag-of-Words
        self.VOCAB = self.BOW.groupby('term_str').tf.sum().to_frame('cf').sort_index() 
        self.VOCAB['df'] = self.BOW.tf.astype(bool).groupby('term_str').sum()
        self.VOCAB['idf'] = np.log2(self.n_docs/self.VOCAB.df)
        self.VOCAB['dfidf'] = self.VOCAB.df * self.VOCAB.idf
        self.VOCAB['term_rank'] = self.VOCAB.cf.rank(method='dense', ascending=False).astype(int)