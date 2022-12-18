from flask import Flask,render_template, request
import numpy as np            
import pickle
import pandas as pd    

model = pickle.load(open('lasso_reg_model.pkl','rb'))

app = Flask(__name__)

@app.route("/")
def my():
    return render_template("home.html" )

@app.route("/predict",methods = ["POST", "GET"])
def home():
    a = request.form['a']
    b = request.form['b']
    c = request.form['c']
    d = request.form['d']
    e = request.form['e']
    f = request.form['f']
    g = request.form['g']
    h = request.form['h']
    i = request.form['i']
    j = request.form['j']
  

    arr = np.array([[a, b, c, d, e, f, g, h, i, j]])
    pred = model.predict(arr)
    return render_template('after.html', prediction = pred)


if __name__ == "__main__":
    app.run(host='0.0.0.0' , port= config1.PORT_NUMBER, debug=True)   

# @app.route('/predict1')
# def predict1():
#     a = int(request.args.get('a'))
#     b = int(request.args.get('b'))
#     c = int(request.args.get('c'))
#     d = int(request.args.get('d'))

#     arr = np.array([[a, b, c, d, e, f, g, h, i, j]])
#     pred = model.predict(arr)
#     return render_template('after.html', data=pred)

    