# Training/Prediction
# Model 1: Using Random Forest

#Import necessary packages and libraries
import numpy as np
import pandas as pd
from nltk.stem.porter import *
stemmer = PorterStemmer()
import re
import random
random.seed(2021)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer


# Import dataset
# dataset is cleaned and structured for easy operations
df_train = pd.read_excel('cord-data.xlsx')  
#df_train = df_train.dropna()    #drops the NA values



def str_stem(s): 
    """
    This function cleans the text in article such as removing stop words, stemming, standalize the cases and so on
    :param s: text / article
    :return: return processed texts
    """ 
    if isinstance(s, str):
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = re.sub(r"([0-9]+)( *)(°|degrees|degree)\.?", r"\1 deg. ", s)
        s = re.sub(r"([0-9]+)( *)(v|volts|volt)\.?", r"\1 volt. ", s)
        s = re.sub(r"([0-9]+)( *)(wattage|watts|watt)\.?", r"\1 watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1 amp. ", s)
        s = re.sub(r"([0-9]+)( *)(qquart|quart)\.?", r"\1 qt. ", s)
        s = re.sub(r"([0-9]+)( *)(hours|hour|hrs.)\.?", r"\1 hr ", s)
        s = re.sub(r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?", r"\1 gal. per min. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r"\1 gal. per hr ", s)
        # Deal with special characters
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("&nbsp;"," ")
        s = s.replace("&amp;","&")
        s = s.replace("&#39;","'")
        s = s.replace("/>/Agt/>","")
        s = s.replace("</a<gt/","")
        s = s.replace("gt/>","")
        s = s.replace("/>","")
        s = s.replace("<br","")
        s = s.replace("<.+?>","")
        s = s.replace("[ &<>)(_,;:!?\+^~@#\$]+"," ")
        s = s.replace("'s\\b","")
        s = s.replace("[']+","")
        s = s.replace("[\"]+","")
        s = s.replace("-"," ")
        s = s.replace("+"," ")
        # Remove text between paranthesis/brackets)
        s = s.replace("[ ]?[[(].+?[])]","")
        # remove sizes
        s = s.replace("size: .+$","")
        s = s.replace("size [0-9]+[.]?[0-9]+\\b","")
        
        
        return " ".join([stemmer.stem(re.sub('[^A-Za-z0-9-./]', ' ', word)) for word in s.lower().split()])
    else:
        return "null"
    
# clean the query term and the doc/article
df_train['query'] = df_train['query'].apply(str_stem)
df_train['document'] = df_train['full_text'].apply(str_stem)



def compute_tf_idf(corpus):
    """
    This function computes the TF-IDF of the article to extract the keywords from the article.
    :param corpus: article/document
    :return: a table of keywords with its tf-idf values 
    """
    
    docs = corpus.split('.')
    
    #instantiate CountVectorizer() 
    cv=CountVectorizer() 

    # this steps generates word counts for the words in your docs 
    word_count_vector=cv.fit_transform(docs)


    word_count_vector.shape

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
    tfidf_transformer.fit(word_count_vector)

    # print idf values 
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"]) 

    # sort ascending 
    df_idf.sort_values(by=['idf_weights'])

    # count matrix 
    count_vector=cv.transform(docs) 

    # tf-idf scores 
    tf_idf_vector=tfidf_transformer.transform(count_vector)

    feature_names = cv.get_feature_names() 
    
    #get tfidf vector for first document 
    first_document_vector=tf_idf_vector[0] 

    
    #print the scores 
    df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"]) 
    res = df.sort_values(by=["tfidf"],ascending=False)

    return res



# process all documents in the dataset
all_corpus = df_train.loc[:,'document']

# Creates new cols of keyword 1 to keyword 5 for each row
for i in range(len(all_corpus)):
    corpus = all_corpus[i]
    if corpus != 'null':
        tf_idf = compute_tf_idf(all_corpus[i])
        features = tf_idf.index[0:5]
        df_train.at[i,'keyword_1'] = features[0]
        df_train.at[i,'keyword_2'] = features[1]
        df_train.at[i,'keyword_3'] = features[2]
        df_train.at[i,'keyword_4'] = features[3]
        df_train.at[i,'keyword_5'] = features[4]



def str_common_word(str1, str2):
    """
    Counts the common word in two strings
    :param str1: a source string
    :param str2: a target string
    :return: the number of times of the common word
    """
    str1, str2 = str1.lower(), str2.lower()
    words, count = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            count+=1
    return count
    
def str_whole_word(str1, str2, i_):
    """
    Counts the whole word from the source string in the target string
    :param str1: a source string
    :param str2: a target string
    :param i_: index of the search
    :return: the number of times of the found whole word
    """
    str1, str2 = str1.lower().strip(), str2.lower().strip()
    count = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return count
        else:
            count += 1
            i_ += len(str1)
    return count


# Drops the rows with NA values
df_train = df_train.dropna()


# Creates new columns of the features of the query term, doc, and from keyword 1 to keyword 5
df_train['word_len_of_query'] = df_train['query'].apply(lambda x:len(x.split())).astype(np.int64)
df_train['word_len_of_document'] = df_train['document'].apply(lambda x:len(x.split())).astype(np.int64)
df_train['word_len_of_kw1'] = df_train['keyword_1'].apply(lambda x:len(str(x).split())).astype(np.int64)
df_train['word_len_of_kw2'] = df_train['keyword_2'].apply(lambda x:len(str(x).split())).astype(np.int64)
df_train['word_len_of_kw3'] = df_train['keyword_3'].apply(lambda x:len(str(x).split())).astype(np.int64)
df_train['word_len_of_kw4'] = df_train['keyword_4'].apply(lambda x:len(str(x).split())).astype(np.int64)
df_train['word_len_of_kw5'] = df_train['keyword_5'].apply(lambda x:len(str(x).split())).astype(np.int64)


# Query & Document
# Create a new column that combine "query" and "total_document" 
df_train['total_info'] = df_train['query']+"\t"+df_train['document'] 
# Number of times the entire search term appears in total document. 
df_train['query_in_document'] = df_train['total_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
# Number of words that appear in search term also appear in total document.
df_train['word_in_document'] = df_train['total_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
# The ratio of total document word length to search term word length
df_train['query_document_len_prop']=df_train['word_len_of_document']/df_train['word_len_of_query']
# The ratio of total document and search term common word count to search term word count
df_train['ratio_document'] = df_train['word_in_document']/df_train['word_len_of_query']




# Keyword 1 & Document
# Create a new column that combine "query" and "total_document" 
df_train['keyword_doc1'] = df_train['keyword_1']+"\t"+df_train['document'] 
# Number of times the entire search term appears in total document. 
df_train['kw1_in_document'] = df_train['keyword_doc1'].apply(lambda x:str_whole_word(str(x).split('\t')[0],str(x).split('\t')[1],0))
# Number of words that appear in search term also appear in total document.
df_train['word_in_document'] = df_train['keyword_doc1'].apply(lambda x:str_common_word(str(x).split('\t')[0],str(x).split('\t')[1]))
# The ratio of total document word length to search term word length
df_train['kw1_document_len_prop']=df_train['word_len_of_document']/df_train['word_len_of_kw1']
# The ratio of total document and search term common word count to search term word count
df_train['kw1_ratio_document'] = df_train['word_in_document']/df_train['word_len_of_kw1']


# Keyword 2 & Document
# Create a new column that combine "query" and "total_document" 
df_train['keyword_doc2'] = df_train['keyword_2']+"\t"+df_train['document'] 
# Number of times the entire search term appears in total document. 
df_train['kw2_in_document'] = df_train['keyword_doc2'].map(lambda x:str_whole_word(str(x).split('\t')[0],str(x).split('\t')[1],0))
# Number of words that appear in search term also appear in total document.
df_train['word_in_document'] = df_train['keyword_doc2'].map(lambda x:str_common_word(str(x).split('\t')[0],str(x).split('\t')[1]))
# The ratio of total document word length to search term word length
df_train['kw2_document_len_prop']=df_train['word_len_of_document']/df_train['word_len_of_kw2']
# The ratio of total document and search term common word count to search term word count
df_train['kw2_ratio_document'] = df_train['word_in_document']/df_train['word_len_of_kw2']


# Keyword 3 & Document
# Create a new column that combine "query" and "total_document" 
df_train['keyword_doc3'] = df_train['keyword_3']+"\t"+df_train['document'] 
# Number of times the entire search term appears in total document. 
df_train['kw3_in_document'] = df_train['keyword_doc3'].map(lambda x:str_whole_word(str(x).split('\t')[0],str(x).split('\t')[1],0))
# Number of words that appear in search term also appear in total document.
df_train['word_in_document'] = df_train['keyword_doc3'].map(lambda x:str_common_word(str(x).split('\t')[0],str(x).split('\t')[1]))
# The ratio of total document word length to search term word length
df_train['kw3_document_len_prop']=df_train['word_len_of_document']/df_train['word_len_of_kw3']
# The ratio of total document and search term common word count to search term word count
df_train['kw3_ratio_document'] = df_train['word_in_document']/df_train['word_len_of_kw3']


# Keyword 4 & Document
# Create a new column that combine "query" and "total_document" 
df_train['keyword_doc4'] = df_train['keyword_4']+"\t"+df_train['document'] 
# Number of times the entire search term appears in total document. 
df_train['kw4_in_document'] = df_train['keyword_doc4'].map(lambda x:str_whole_word(str(x).split('\t')[0],str(x).split('\t')[1],0))
# Number of words that appear in search term also appear in total document.
df_train['word_in_document'] = df_train['keyword_doc4'].map(lambda x:str_common_word(str(x).split('\t')[0],str(x).split('\t')[1]))
# The ratio of total document word length to search term word length
df_train['kw4_document_len_prop']=df_train['word_len_of_document']/df_train['word_len_of_kw4']
# The ratio of total document and search term common word count to search term word count
df_train['kw4_ratio_document'] = df_train['word_in_document']/df_train['word_len_of_kw4']


# Keyword 5 & Document
# Create a new column that combine "query" and "total_document" 
df_train['keyword_doc5'] = df_train['keyword_5']+"\t"+df_train['document'] 
# Number of times the entire search term appears in total document. 
df_train['kw5_in_document'] = df_train['keyword_doc5'].map(lambda x:str_whole_word(str(x).split('\t')[0],str(x).split('\t')[1],0))
# Number of words that appear in search term also appear in total document.
df_train['word_in_document'] = df_train['keyword_doc5'].map(lambda x:str_common_word(str(x).split('\t')[0],str(x).split('\t')[1]))
# The ratio of total document word length to search term word length
df_train['kw5_document_len_prop']=df_train['word_len_of_document']/df_train['word_len_of_kw5']
# The ratio of total document and search term common word count to search term word count
df_train['kw5_ratio_document'] = df_train['word_in_document']/df_train['word_len_of_kw5']



df_train.drop(['query','round','doc_id','full_text','document','total_info','keyword_1','keyword_2','keyword_3','keyword_4','keyword_5','query_in_document', 'kw1_ratio_document','kw2_ratio_document','kw3_ratio_document','kw4_ratio_document','kw5_ratio_document','keyword_doc1','keyword_doc2','keyword_doc3','keyword_doc4','keyword_doc5'], axis=1, inplace=True)



# Separate the relevancy col and the rest of features cols
X = df_train.loc[:, df_train.columns != 'relevancy']
y = df_train.loc[:, df_train.columns == 'relevancy']

# print(X)


# Encode the features using one hot encoder since there's strings features
# temp = OneHotEncoder().fit_transform(X).toarray()
# X = temp

# print(X)
# print(len(X[0]))

# Splits the data into train and test sets with 70% and 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fits the data to a random forest
rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=0)
rf.fit(X_train, y_train.values.ravel())


pickle.dump(rf, open("ml_model.sav", "wb"))
