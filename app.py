from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Time=float(request.form.get('Time')),
            Ipv=float(request.form.get('Ipv')),
            Vpv=float(request.form.get('Vpv')),
            Vdc=float(request.form.get('Vdc')),
            ia=float(request.form.get('ia')),
            ib=float(request.form.get('ib')),
            ic=float(request.form.get('ic')),
            va=float(request.form.get('va')),
            vb=float(request.form.get('vb')),
            vc=float(request.form.get('vc')),
            Iabc=float(request.form.get('Iabc')),
            If=float(request.form.get('If')),
            Vabc=float(request.form.get('Vabc')),
            Vf=float(request.form.get('Vf'))
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template('home.html', results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)



