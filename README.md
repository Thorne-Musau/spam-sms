## SMS Spam Detector

This is a machine learning project that uses a pre-trained ML model with the ***spam.csv*** dataset 
The model is then loaded onto a web app that can be used to detect whether a message input is spam or not

### Tools
- Python
- Pandas
- Numpy
- Matplotlib
- Frontend - HTML & CSS
- Flask

--- 

## Project flow

### Setting up libraries
I first set up the dependancies needed for the project on the virtual environment. That includes, flask, jupyter, matplotlib  etc

### sms_spam.ipynb
We then create the detection model using the spam.csv dataset. Perform some data cleaning, feature 
engineering and visualization of the spam.csv data
We then use the stemmer machine algorithm to create and train a model

### Frontend
I then used Flask in conjuction with HTML and CSS to create the web app that I embedded the machine learning model save in the .pkl files