from flask import Flask, render_template, request
import numpy as np
import pickle
trained_model = pickle.load(open('logit_model_iris.pkl','rb'))
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def basic():
    if request.method == 'POST':
        sepal_length = request.form['sepallength']
        sepal_width = request.form['petalwidth']
        petal_length = request.form['petallength']
        petal_width = request.form['petalwidth']
        y_pred = [[sepal_length, sepal_width, petal_length, petal_width]]
        y_pred = np.array(y_pred,dtype = float)
        prediction_value = trained_model.predict(y_pred)

        setosa = 'The flower is classified as "Iris-Setosa"'
        versicolor = 'The flower is classified as "Iris-Versicolor"'
        virginica = 'The flower is classified as "Iris-Virginica"'
        if prediction_value == 0:
            return render_template('index.html', setosa=setosa)
        elif prediction_value == 1:
            return render_template('index.html', versicolor=versicolor)
        elif prediction_value == 2:
            return render_template('index.html', virginica=virginica) 
        else:
            return render_template('index.html', result = "invalid") 
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)