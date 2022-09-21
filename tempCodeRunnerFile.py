from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import naive_bayes, linear_model, svm, ensemble
import xgboost
from feature_engineering import X_data_tfidf_ngram, y_data_n

def train_model(classifier, classifier_name ,X_data, y_data):       
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

    classifier.fit(X_train, y_train)
    
    test_predictions = classifier.predict(X_test)

    acc = metrics.accuracy_score(y_test, test_predictions)
    pre = metrics.precision_score(y_test, test_predictions,average='micro')
    rec = metrics.recall_score(y_test, test_predictions,average='micro')
    f1 = metrics.f1_score(y_test, test_predictions,average='micro')

    result = 'Danh gia thuat toan ' + classifier_name + '\n'
    result += "Accuracy: " + str(acc) + '\n'
    result += "Precision: " + str(pre) + '\n'
    result += "Recall: " + str(rec) + '\n'
    result += "F1: " + str(f1) + '\n'
    result += '----------------------------------------------------------------------'
    print(result)