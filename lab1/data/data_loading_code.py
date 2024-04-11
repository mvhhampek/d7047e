import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
nltk.download('punkt')
nltk.download('stopwords')


def preprocess_pandas(data, columns):
    df_ = pd.DataFrame(columns=columns)
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].str.replace(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
    data['Sentence'] = data['Sentence'].str.replace(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
    data['Sentence'] = data['Sentence'].str.replace(r'[^\w\s]','')                                                       # remove special characters
    data['Sentence'] = data['Sentence'].replace('\d', '', regex=True)                                                   # remove numbers
    for index, row in data.iterrows():
        word_tokens = word_tokenize(row['Sentence'])
        filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
        df_ = df_._append({
            "index": row['index'],
            "Class": row['Class'],
            "Sentence": " ".join(filtered_sent[0:])
        }, ignore_index=True)
    return data


def get_data():
    data = pd.read_csv("lab1/data/amazon_cells_labelled.txt", delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index                                          # add new column index
    columns = ['index', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns)                             # pre-process
    training_data, validation_data, training_labels, validation_labels = train_test_split( # split the data into training, validation, and test splits
        data['Sentence'].values.astype('U'),
        data['Class'].values.astype('int32'),
        test_size=0.10,
        random_state=0,
        shuffle=True
    )


    # vectorize data using TFIDF and transform for PyTorch for scalability
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=50000, max_df=0.5, use_idf=True, norm='l2')
    training_data = word_vectorizer.fit_transform(training_data)        # transform texts to sparse matrix
    training_data = training_data.todense()                             # convert to dense matrix for Pytorch
    vocab_size = len(word_vectorizer.vocabulary_)
    validation_data = word_vectorizer.transform(validation_data)
    validation_data = validation_data.todense()
    train_x_tensor = torch.from_numpy(np.array(training_data)).type(torch.FloatTensor)
    train_y_tensor = torch.from_numpy(np.array(training_labels)).long()
    validation_x_tensor = torch.from_numpy(np.array(validation_data)).type(torch.FloatTensor)
    validation_y_tensor = torch.from_numpy(np.array(validation_labels)).long()

    # Prints to confirm the data is loaded and pre-processed
    print("Training data: ", train_x_tensor.shape)
    print("Training labels: ", train_y_tensor.shape)
    print("Validation data: ", validation_x_tensor.shape)
    print("Validation labels: ", validation_y_tensor.shape)
    print("Vocabulary size: ", vocab_size)
    print("Data loaded and pre-processed successfully")

    return train_x_tensor, train_y_tensor, validation_x_tensor, validation_y_tensor, vocab_size, word_vectorizer

def get_data_test():
    # get data, pre-process and split
    # data = pd.read_csv("/home/convergent/PycharmProjects/Labs D7047E/Lab 1/data/amazon_cells_labelled.txt", delimiter='\t', header=None)
    data = pd.read_csv("lab1/data/amazon_cells_labelled.txt", delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index                                          # add new column index
    columns = ['index', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns)                             # pre-process
    intermediate_data, test_data, intermediate_labels, test_labels = train_test_split( # split the data into training, validation, and test splits
        data['Sentence'].values.astype('U'),
        data['Class'].values.astype('int32'),
        test_size=0.20,
        random_state=0,
        shuffle=True
    )
    training_data, validation_data, training_labels, validation_labels = train_test_split( # split the data into training, validation, and test splits
        intermediate_data,
        intermediate_labels,
        test_size=0.25,
        random_state=0,
        shuffle=True
    )


    # vectorize data using TFIDF and transform for PyTorch for scalability
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=50000, max_df=0.5, use_idf=True, norm='l2')
    training_data = word_vectorizer.fit_transform(training_data)        # transform texts to sparse matrix
    training_data = training_data.todense()                             # convert to dense matrix for Pytorch
    vocab_size = len(word_vectorizer.vocabulary_)
    
    validation_data = word_vectorizer.transform(validation_data)
    validation_data = validation_data.todense()
    
    test_data = word_vectorizer.transform(test_data)
    test_data = test_data.todense()
    
    train_x_tensor = torch.from_numpy(np.array(training_data)).type(torch.FloatTensor)
    train_y_tensor = torch.from_numpy(np.array(training_labels)).long()
    
    validation_x_tensor = torch.from_numpy(np.array(validation_data)).type(torch.FloatTensor)
    validation_y_tensor = torch.from_numpy(np.array(validation_labels)).long()
    
    test_x_tensor = torch.from_numpy(np.array(test_data)).type(torch.FloatTensor)
    test_y_tensor = torch.from_numpy(np.array(test_labels)).long()

    # Prints to confirm the data is loaded and pre-processed
    print("Training data: ", train_x_tensor.shape)
    print("Training labels: ", train_y_tensor.shape)
    print("Validation data: ", validation_x_tensor.shape)
    print("Validation labels: ", validation_y_tensor.shape)
    print("Test data: ", test_x_tensor.shape)
    print("Test labels: ", test_y_tensor.shape)
    print("Vocabulary size: ", vocab_size)
    print("Data loaded and pre-processed successfully")

    return train_x_tensor, train_y_tensor, validation_x_tensor, validation_y_tensor, vocab_size, word_vectorizer, test_x_tensor, test_y_tensor



# If this is the primary file that is executed (ie not an import of another file)
if __name__ == "__main__":
    
    train_x_tensor, train_y_tensor, validation_x_tensor, validation_y_tensor, vocab_size, word_vectorizer = get_data()