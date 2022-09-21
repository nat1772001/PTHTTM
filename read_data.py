import csv
from pyvi import ViTokenizer # thư viện NLP tiếng Việt
import gensim # thư viện NLP
import pickle

def get_data(learning_file):
    X = []
    y = []
    with open(learning_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            data = row[0] + '\n' +row[1]
            data = gensim.utils.simple_preprocess(data) # xoa cac ki tu khoang trong va cac ki tu dac biet
            print(data)
            data = ' '.join(data)
            data = ViTokenizer.tokenize(data) # tach tu co nghia
            X.append(data)
            y.append(row[2])
    return X, y

def stopword_eliminate(data):
    return data

train_file = 'dataset.csv'
X_data, y_data = get_data(train_file)  
pickle.dump(X_data, open('data/X_data.pkl', 'wb'))
pickle.dump(y_data, open('data/y_data.pkl', 'wb'))