import nltk
from nltk.corpus import wordnet
import spacy 
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model
import re
import string
from pandas.api.types import CategoricalDtype 
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel, BertConfig
import bert
 
#from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.regularizers import l2
from keras.layers import Flatten
from keras.layers import Dense, Input , Dropout
from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
import spacy 
import textstat
from textstat.textstat import textstatistics,legacy_round 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')

pos_dict = {'DET': 1, 'NOUN': 2, 'AUX': 3,
	    'PROPN': 4, 'CCONJ': 5, 'PRON': 6, 'PUNCT': 7}
nlp = spacy.load('en')

low_freq_characters = ['f', 'h', 'j', 'k', 'q', 'w', 'x', 'z']
low_freq_dict = {'f': 0.308, 'h': 0.297, 'j': 0.507,
    'k': 0.989, 'q': 0.123, 'w': 0.983, 'x': 0.785, 'z': 0.533}
high_freq_characters = ['e', 'a', 'o', 's', 'r']
high_freq_dict = {'e': 1.218, 'a': 1.153, 'o': 0.868, 's': 0.798, 'r': 0.687}
max_seq_length=512
max_seq_length_context=210


class Features_extracter(object):

	
	def __init__(self):
		self.bert_model_sequence = None
		self.bert_model_pooled = None
		self.bert_tokenizer = None
		self.embeddings_index = {}

	def clean_data(self, raw_data):
		raw_data = raw_data.lower()
		raw_data = raw_data.strip()
		cleanr = re.compile('['+string.punctuation+']')
		cleantext = re.sub(cleanr, '', raw_data)
		return cleantext

	def tokn_len(self, tkn):
		return len(tkn)

	def isvowel(self, c):
		if (c == 'a' or c == 'e' or c == 'i' or c == 'o' or c == 'u'):
			return True
		return False

	def num_vowels(self, sen):
		count = 0
		for c in sen:
			if self.isvowel(c):
	  			count += 1

		return count

	def pos_tag_token(self, df):
		pos_tag = []
		det_c_li = []
		noun_c_li = []
		pron_c_li = []
		verb_c_li = []
		adj_c_li = []
		cconj_c_li = []
		adv_c_li = []
		for index, row in df.iterrows():
			doc = nlp(str(row['sentence']))
			det_c = 0
			noun_c = 0
			adj_c = 0
			propn_c = 0
			cconj_c = 0
			pron_c = 0
			punct_c = 0
			verb_c = 0
			adv_c = 0

			not_found = 1
			for w in doc:
				if (w.pos_ == 'DET'):
					det_c += 1

				if (w.pos_ == 'NOUN'):
					noun_c += 1

				if (w.pos_ == 'PRON'):
					pron_c += 1

				if (w.pos_ == 'VERB'):
					verb_c += 1

				if (w.pos_ == 'ADJ'):
					adj_c += 1

				if (w.pos_ == 'CCONJ'):
					cconj_c += 1

				if (w.pos_ == 'ADV'):
					adv_c += 1

				if (w.lower_ == (str(row['token']).lower()) and (not_found == 1)):
					if w.pos_ in pos_dict:
						pos_tag.append(pos_dict[w.pos_])
					else:
						pos_tag.append(0)
					not_found = 0

			if (not_found == 1):
				pos_tag.append(0)

			det_c_li.append(det_c)
			noun_c_li.append(noun_c)
			pron_c_li.append(pron_c)
			verb_c_li.append(verb_c)
			adj_c_li.append(adj_c)
			cconj_c_li.append(cconj_c)
			adv_c_li.append(adv_c)

		df['pos_tag'] = pos_tag
		df['det_pos'] = det_c_li
		df['noun_pos']= noun_c_li
		df['pron_pos']= pron_c_li
		df['verb_pos']= verb_c_li
		df['adj_pos'] =  adj_c_li
		df['cconj_pos']= cconj_c_li
		df['adv_pos']= adv_c_li
		return df






	def high_freq_score(self, txt):
		score=0
		for c in txt:
			if c in high_freq_characters:
				score+=high_freq_dict[c]
		return score


	def low_freq_score(self, txt):
		score=0
		for c in txt:
			if c in low_freq_characters:
				score+=low_freq_dict[c]
		return score



	# synonyms
	def synonyms(self, word):
	  synonyms = [] 
	  for syn in wordnet.synsets(word): 
		  for l in syn.lemmas(): 
		      synonyms.append(l.name()) 
	  
	  synonyms = set(map(lambda x: self.clean_data(x), synonyms))
		
	  return len(synonyms)



	# antonyms
	def antonyms(self, word):
	  antonyms = [] 
	  for syn in wordnet.synsets(word): 
		  for l in syn.lemmas(): 
		      if l.antonyms(): 
		          antonyms.append(l.antonyms()[0].name()) 
	  
	  antonyms = set(map(lambda x: self.clean_data(x), antonyms))    
	  return len(antonyms)





	def hypernyms(self, word):
		hypernyms=0
		try:
		    results = wordnet.synsets(word)
		    hypernyms = len(results[0].hypernyms())
		    return hypernyms
		except:
		    return hypernyms




	def hyponyms(self, word):
		hyponyms=0
		try:
		    results = wordnet.synsets(word)
		except:
		    return hyponyms
		try:
		    hyponyms = len(results[0].hyponyms())
		    return hyponyms
		except:
		    return hyponyms


	def map_pos(self, df):
		positions = []
		for index, row in df.iterrows():
			positions.append(str(row['sentence']).find(str(row['token'])))
		df['positions'] = positions
		return df



	def get_wordlen(self, x):
		return len(x.split())


	def split_txt(self, txt):
		return txt.split()




	def token_to_id(self, k):
		lis=[]
		for p in k:
			lis.append(self.bert_tokenizer.vocab[p])
		return lis


	
	
	def modify_tokens(self, tkns):
		# print((tkns))
		tkns.insert(0,'[CLS]')
		l=len(tkns)
		
		if (l>max_seq_length-1):
			tkns=tkns[:max_seq_length-1]

		mask=[0]*max_seq_length

		for i in range(min(l,max_seq_length)):
			mask[i]=1

		for i in range(max_seq_length-1):
			if(mask[i]==0):
				tkns.append('[PAD]')
		    
		

		tkns.append('[SEP]')

		mask[max_seq_length-1]=1
		
		s1=0
		s2=0  
    

		return tkns
	


	def mask(self, tkns):

		msk=[0]*max_seq_length
	   
		for i in range(len(tkns)):
			if tkns[i]=='[PAD]':
				msk[i]=0
			else:
				msk[i]=1
		msk[max_seq_length-1]=1  
		

		return msk



	def segment(self, tkns):
		segment=[0]*max_seq_length 
		return segment

	
	
	def convert_int(self, li):
		ans=[]
		for i in li:
			ans.append(int(i))

		return ans


	# ---------------------------------------------------------
	# ------------------find sentense wise glove embeddings-----------------
	# ---------------------------------------------------------

	def glove_vec_for_sentence(self, data):
		allSent_glov_vect = []
		glove_embeddings_index = self.embeddings_index
		for sent in data:
			glob_vector_train_sentence=[]
			for word in sent.split(' '):
				if word in glove_embeddings_index:
					glob_vector_train_sentence.append(glove_embeddings_index[word])
				else:
					glob_vector_train_sentence.append(np.zeros(300,))


			glob_vector_train_sentence=np.array(glob_vector_train_sentence)
			glob_vector_train_sentence_vec = glob_vector_train_sentence.mean(axis=0)
			allSent_glov_vect.append(glob_vector_train_sentence_vec)

		allSent_glov_vect = np.array(allSent_glov_vect)
		print(allSent_glov_vect.shape)
		return allSent_glov_vect



	def gloveEmbedding(self):
		f = open('glove.42B.300d.txt', encoding='utf-8')
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			self.embeddings_index[word] = coefs
		f.close()




	
# ________________________________________________________________________	
	
	def context_info(self, df):
		context_texts=[]
		for index, row in df.iterrows(): 
			sentence    =   str(row['sentence'])
			target_word =   str(row['token'])

			sentence_words=  sentence.split(" ")
			sentence_len =   len(sentence_words)
			
			target_word_pos=0
			for i in sentence_words:
				if (i==target_word):
					break
				else:
					target_word_pos+=1

			context_txt=""
			
			for i in range(max(0,target_word_pos-10),target_word_pos):
				context_txt+=sentence_words[i]+" "

			for i in range(target_word_pos,min(len(sentence_words),10+target_word_pos)):
				context_txt+=sentence_words[i]+" "

			
			context_texts.append(context_txt)
		
		df['context']=context_texts

		return df

	
	
	
	
	
# ----------------------------------------------------------------------------	
	
	

	
	def preprocessing(self, df):
		self.gloveEmbedding()
		
		df = self.pos_tag_token(df)
		
		df['sentence']=df['sentence'].map(self.clean_data)

		df['token']=df['token'].map(str)

		df['token']=df['token'].map(self.clean_data)

		df['tkn_len']=df['token'].map(self.tokn_len)

		df['num_vowel']=df['sentence'].map(self.num_vowels)

		df['high_freq_score']=df['sentence'].map(self.high_freq_score)

		df['low_freq_score']=df['sentence'].map(self.low_freq_score)

		df['tkn_synonyms'] = df['token'].map(self.synonyms)
		df['tkn_antonyms'] = df['token'].map(self.antonyms)
		df['tkn_hypernyms'] = df['token'].map(self.hypernyms)
		df['tkn_hyponyms'] = df['token'].map(self.hyponyms)

		df=pd.concat([df,pd.get_dummies(df["corpus"],prefix='corpus',drop_first=False)],axis=1)
		df.drop(['corpus','sentence','token'],axis=1)
		df = self.map_pos(df)
		df['len']=df['sentence'].map(self.get_wordlen)

		# y=df['complexity']
		x=df
		
		self.bert_wala()
		
		#x=x['sentence'].map(self.bert_tokenizer.tokenize)
		x_tokens=x['sentence'].map(self.bert_tokenizer.tokenize)
		x_tkns = x_tokens

		
		x_tkns=x_tkns.map(self.modify_tokens)
		x_tkns_id=x_tkns.map(self.token_to_id)
		x_tkns_id=x_tkns_id.map(self.convert_int)
		x_tkns_mask=x_tkns.map(self.mask)
		
		
		x_tkns_segment=np.zeros((x_tkns.shape[0],max_seq_length))
		x_tkns=np.vstack(x_tkns_id)
		x_tkns_mask=np.vstack(x_tkns_mask)
		x_tkns_segment=np.vstack(x_tkns_segment)
		
		tokens=x['token']
		tok=list(tokens)
		
		context, number_of_splits = self.predict_in_batch(x, 32, [x_tkns, x_tkns_mask, x_tkns_segment], list(x_tokens), tok)

		
		
		predict_pooled = self.bert_model_pooled.predict([x_tkns,x_tkns_mask,x_tkns_segment])

		#predict=self.bert_model.predict([x_tkns,x_tkns_mask,x_tkns_segment])

		
		glob_vector_tokens = self.glove_vec_for_sentence(x['token'])
		glob_vector_sentence = self.glove_vec_for_sentence(x['sentence'])



		tlen=x['len'].to_numpy().reshape(-1,1)
		pos=x['positions'].to_numpy().reshape(-1,1)
		corpus_bible=x['corpus_bible'].to_numpy().reshape(-1,1)
		corpus_biomed=x['corpus_biomed'].to_numpy().reshape(-1,1)
		corpus_europarl=x['corpus_europarl'].to_numpy().reshape(-1,1)
		token_len=x['tkn_len'].to_numpy().reshape(-1,1)
		num_vowel=x['num_vowel'].to_numpy().reshape(-1,1)
		pos_tag=x['pos_tag'].to_numpy().reshape(-1,1)
		det_pos=x['det_pos'].to_numpy().reshape(-1,1)
		noun_pos=x['noun_pos'].to_numpy().reshape(-1,1)
		pron_pos=x['pron_pos'].to_numpy().reshape(-1,1)
		verb_pos=x['verb_pos'].to_numpy().reshape(-1,1)
		adj_pos=x['adj_pos'].to_numpy().reshape(-1,1)
		cconj_pos=x['cconj_pos'].to_numpy().reshape(-1,1)
		adv_pos=x['adv_pos'].to_numpy().reshape(-1,1)
		high_freq=x['high_freq_score'].to_numpy().reshape(-1,1)
		low_freq=x['low_freq_score'].to_numpy().reshape(-1,1)
		# tkn_synonyms=x['tkn_synonyms'].to_numpy().reshape(-1,1)
		# tkn_antonyms=x['tkn_antonyms'].to_numpy().reshape(-1,1)
		# tkn_hypernyms=x['tkn_hypernyms'].to_numpy().reshape(-1,1)
		# tkn_hyponyms=x['tkn_hyponyms'].to_numpy().reshape(-1,1)
		# tkn_split=number_of_splits.to_numpy().reshape(-1,1)
		# tkn_split = np.array(number_of_splits).reshape(-1,1)
		feature_vector = np.hstack ((predict_pooled,context, glob_vector_tokens, glob_vector_sentence, tlen,pos,corpus_bible, corpus_biomed, corpus_europarl, token_len, num_vowel,pos_tag,det_pos,noun_pos,pron_pos,verb_pos,adj_pos,cconj_pos,adv_pos,high_freq,low_freq))
		
		
		
		return feature_vector


	
	
	
	
	def preprocessing_m(self, df):
		self.gloveEmbedding()
		
		df = self.pos_tag_token(df)
		
		df['sentence']=df['sentence'].map(self.clean_data)

		df['token']=df['token'].map(str)

		df['token']=df['token'].map(self.clean_data)

		df['tkn_len']=df['token'].map(self.tokn_len)

		df['num_vowel']=df['sentence'].map(self.num_vowels)

		df['high_freq_score']=df['sentence'].map(self.high_freq_score)

		df['low_freq_score']=df['sentence'].map(self.low_freq_score)

		df['tkn_synonyms'] = df['token'].map(self.synonyms)
		df['tkn_antonyms'] = df['token'].map(self.antonyms)
		df['tkn_hypernyms'] = df['token'].map(self.hypernyms)
		df['tkn_hyponyms'] = df['token'].map(self.hyponyms)

		df=pd.concat([df,pd.get_dummies(df["corpus"],prefix='corpus',drop_first=False)],axis=1)
		df.drop(['corpus','sentence','token'],axis=1)
		df = self.map_pos(df)
		df['len']=df['sentence'].map(self.get_wordlen)

		#y=df['complexity']
		x=df
		
		self.bert_wala()
		
		#x=x['sentence'].map(self.bert_tokenizer.tokenize)
		x_tkns=x['sentence'].map(self.split_txt)
		
		x_tkns2=[]
		for tokens in x_tkns:
			tmp=[]
		  
			for i in tokens:
				if i in self.bert_tokenizer.vocab:
					tmp.append(i)

			x_tkns2.append(tmp)
		
		
		x_tkns2=np.array(x_tkns2)
		x_tkns2=pd.DataFrame(x_tkns2)
		
		x_tkns2=x_tkns2[0].map(self.modify_tokens)
		x_tkns2_id=x_tkns2.map(self.token_to_id)
		x_tkns2_id=x_tkns2_id.map(self.convert_int)
		x_tkns2_mask=x_tkns2.map(self.mask)
		
		
		x_tkns2_segment=np.zeros((x_tkns2.shape[0],max_seq_length))
		x_tkns2=np.vstack(x_tkns2_id)
		x_tkns2_mask=np.vstack(x_tkns2_mask)
		x_tkns2_segment=np.vstack(x_tkns2_segment)
		
		predict=self.bert_model_pooled.predict([x_tkns2,x_tkns2_mask,x_tkns2_segment])

		
		glob_vector_tokens = self.glove_vec_for_sentence(x['token'])
		glob_vector_sentence = self.glove_vec_for_sentence(x['sentence'])



		tlen=x['len'].to_numpy().reshape(-1,1)
		pos=x['positions'].to_numpy().reshape(-1,1)
		corpus_bible=x['corpus_bible'].to_numpy().reshape(-1,1)
		corpus_biomed=x['corpus_biomed'].to_numpy().reshape(-1,1)
		corpus_europarl=x['corpus_europarl'].to_numpy().reshape(-1,1)
		token_len=x['tkn_len'].to_numpy().reshape(-1,1)
		num_vowel=x['num_vowel'].to_numpy().reshape(-1,1)
		pos_tag=x['pos_tag'].to_numpy().reshape(-1,1)
		det_pos=x['det_pos'].to_numpy().reshape(-1,1)
		noun_pos=x['noun_pos'].to_numpy().reshape(-1,1)
		pron_pos=x['pron_pos'].to_numpy().reshape(-1,1)
		verb_pos=x['verb_pos'].to_numpy().reshape(-1,1)
		adj_pos=x['adj_pos'].to_numpy().reshape(-1,1)
		cconj_pos=x['cconj_pos'].to_numpy().reshape(-1,1)
		adv_pos=x['adv_pos'].to_numpy().reshape(-1,1)
		high_freq=x['high_freq_score'].to_numpy().reshape(-1,1)
		low_freq=x['low_freq_score'].to_numpy().reshape(-1,1)
		#tkn_synonyms=x_train['tkn_synonyms'].to_numpy().reshape(-1,1)
		#tkn_antonyms=x_train['tkn_antonyms'].to_numpy().reshape(-1,1)
		#tkn_hypernyms=x_train['tkn_hypernyms'].to_numpy().reshape(-1,1)
		#tkn_hyponyms=x_train['tkn_hyponyms'].to_numpy().reshape(-1,1)

		feature_vector = np.hstack ((predict, glob_vector_tokens, glob_vector_sentence, tlen,pos,corpus_bible, corpus_biomed, corpus_europarl, token_len, num_vowel,pos_tag,det_pos,noun_pos,pron_pos,verb_pos,adj_pos,cconj_pos,adv_pos,high_freq,low_freq))
		
		
		
		return feature_vector


	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	def bert_wala(self):
		# Loading the Pretrained Model from tensorflow HUB
		tf.keras.backend.clear_session()

		max_seq_length=512


		# this is input words. Sequence of words represented as integers
		input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")

		# mask vector if you are padding anything
		input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")


		segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")

		# bert layer 
		bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True)
		pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

		# Bert model
		
		self.bert_model_sequence = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=sequence_output)

		self.bert_model_pooled = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=pooled_output)

		
		vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
		do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

		self.bert_tokenizer = BertTokenizer(vocab_file,do_lower_case)

	
	
#-----------------------------	
	def create_context_vector(self,start,end,sequence_output , train_tokens,target_tokens,number_of_splits):
		context_vec=""
		for i in range(end-start):
		    tokn=target_tokens[start+i]
		    train_tkns=list(train_tokens[start+i])
		    bert_output=sequence_output[i]

		    tmp_vector=np.zeros(768)
		    tmp_str=""
		    add_flag=0
		    c=0
		    for j in range(len(train_tkns)):
		        tmp_str+=re.sub("#","",train_tkns[j])

		        #print(tmp_str)
		        if (tokn.startswith(tmp_str)):
		            add_flag=1
		            #print(tmp_str,train_tkns[j],train_tkns[j+1])

		        else:
		            if (add_flag==1):
		                break
		            add_flag=0
		            tmp_str=""

		        if (add_flag==1):
		            c+=1
		            tmp_vector=np.add(tmp_vector,bert_output[j])
		
		    if (i==0):    
		      if (c!=0):
		          context_vec=(tmp_vector/c)
		      else:
		          context_vec=(tmp_vector)
		    else:
		      if (c!=0):
		        context_vec=np.vstack((context_vec,tmp_vector/c))
		      else:
		        context_vec=np.vstack((context_vec,tmp_vector))

		    number_of_splits.append(c)
		    

		
		return context_vec

	
	
	
	
	
	def predict_in_batch(self,df,batch_size,vector,sen_tokens,tok):
		l=vector[0].shape[0]
		number_of_splits=[]
		predict_vec=""
		for i in range(batch_size,l+1,batch_size):    
			tkns=vector[0]
			msk=vector[1]
			seg=vector[2]
			indx=[j for j in range(i-batch_size,i)]
			tkns=tkns[indx,:]
			msk=msk[indx,:]
			seg=seg[indx,:]

			predt=self.bert_model_sequence.predict([tkns,msk,seg])

			predt=self.create_context_vector(i-batch_size,i,predt,sen_tokens,tok,number_of_splits)
			
			if (i==batch_size):
				predict_vec=predt
			else:
				predict_vec=np.vstack((predict_vec,predt))


		batch_size=l%batch_size
		tkns=vector[0]
		msk=vector[1]
		seg=vector[2]

		indx=[j for j in range(l-batch_size,l)]
		tkns=tkns[indx,:]
		msk=msk[indx,:]
		seg=seg[indx,:]

		predt=self.bert_model_sequence.predict([tkns,msk,seg])
		predt=self.create_context_vector(l-batch_size,l,predt,sen_tokens,tok,number_of_splits)
		predict_vec=np.vstack((predict_vec,predt))

		#df['token_splits']=number_of_splits

		print(predict_vec.shape)
		return predict_vec,number_of_splits
	
	
	
	
	
	
	
