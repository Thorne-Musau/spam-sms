from flask import Flask, render_template, request, session
import os
import joblib
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/spam_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open("model\spam_model.pkl", "rb"))
    tfidf_model = pickle.load(open("model/tfidf_model.pkl", "rb"))
    if request.method == "POST":
        message = session.get('message')
        message =[message]
        dataset ={'message': message}
        data =pd.DataFrame(dataset)
        data["message"] = data["message"].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')
        data["message"] = data["message"].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')
        data["message"] = data["message"].str.replace(r'£|\$', 'money-symbol')
        data["message"] = data["message"].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phone-number')
        data["message"] = data["message"].str.replace(r'\d+(\.\d+)?', 'number')
        data["message"] = data["message"].str.replace(r'[^\w\d\s]', ' ')
        data["message"] = data["message"].str.replace(r'\s+', ' ')
        data["message"] = data["message"].str.replace(r'^\s+|\s*?$', ' ')
        data["message"] = data["message"].str.lower()

        stop_words = set(stopwords.words('english'))
        data["message"] = data["message"].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
        ss = nltk.SnowballStemmer("english")
        data["message"] = data["message"].apply(lambda x: ' '.join(ss.stem(term) for term in x.split()))

        # tfidf_model = TfidfVectorizer()
        tfidf_vec = tfidf_model.transform(data["message"])
        tfidf_data = pd.DataFrame(tfidf_vec.toarray())
        my_prediction = model.predict(tfidf_data)
        
    return render_template('predict.html', prediction=my_prediction)


    
          


if __name__ == '__main__':
    app.run(debug=True)
