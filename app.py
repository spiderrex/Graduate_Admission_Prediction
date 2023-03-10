from flask import Flask, render_template, request
import numpy as np
import pickle  
# Load the trained model
model = pickle.load(open('admission_prediction_model.pkl','rb'))

# Define the Flask app
app = Flask(__name__)

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')


# Define the prediction route
@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
       gre_score = int(request.form['GRE Score'])
       toefl_score = int(request.form['TOEFL Score'])
       university_rating = int(request.form['University Rating'])
       sop_score = float(request.form['SOP'])
       lor_score = float(request.form['LOR'])
       cgpa = float(request.form['CGPA'])
       research = int(request.form['Research'])
       
       
       prediction = model.predict([[gre_score, toefl_score, university_rating, sop_score, lor_score, cgpa, research]])
       
       if prediction == 1:
        prediction_text = 'This student is likely to be admitted.'
       else:
        prediction_text = 'This student is unlikely to be admitted.'
          
       return render_template('prediction.html',prediction_text="Prediction :{}".format(prediction_text))    

    
    else:
        return render_template('prediction.html')
        
   

   
    

if __name__ == '__main__':
    app.run(debug=True)
