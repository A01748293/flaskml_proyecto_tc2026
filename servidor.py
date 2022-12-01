from flask import Flask, request, jsonify, render_template
import numpy as np
from joblib import load, dump
from werkzeug.utils import secure_filename
import os
from os import path
import json
import requests
from flask_cors import CORS, cross_origin
#Extracción de datos
import pandas as pd
#Biblioteca de manejo de vectores y matrices
import numpy as np
#Biblioteca para exportar procesos de machine learning

#Cargar el modelo
# if (path.exists('modelo.joblib')):
dt = load('modelo.joblib')

#Generar el servidor (Back-end)
servidorWeb = Flask(__name__)
CORS(servidorWeb)
#Envio de datos a través de Archivos
@servidorWeb.route('/modeloFile', methods=['POST'])
def modeloFile():
    f = request.files['file']
    filename=secure_filename(f.filename)
    path=os.path.join(os.getcwd(),'files',filename)
    f.save(path)
    file = open(path, "r")
    
    for x in file:
        info=x.split()
    print(info)
    datosEntrada = np.array([
            float(info[0]),
            float(info[1]),
            float(info[2]),
            float(info[3]),
            float(info[4]),
            float(info[5]),
            float(info[6]),
            float(info[7]),
            float(info[8])
        ])
    #Utilizar el modelo
    resultado=dt.predict(datosEntrada.reshape(1,-1))
    #Regresar la salida del modelo
    return jsonify({"Resultado":str(resultado[0])})

#Envio de datos a través de Forms
@servidorWeb.route('/modeloFormDiabetes', methods=['POST'])
def modeloForm():
    #Procesar datos de entrada 
    contenido = request.form
    
    datosEntrada = np.array([
            contenido['pregnancies'],
            contenido['glucose'],
            contenido['bloodPressure'],
            contenido['skinThickness'],
            contenido['insulin'],
            contenido['BMI'],
            contenido['DPF'],
            contenido['age']
        ])
    #Utilizar el modelo
    resultado=dt.predict(datosEntrada.reshape(1,-1))
    #Regresar la salida del modelo
    return jsonify({"Resultado":str(resultado[0])})


#Envio de datos a través de JSON
@servidorWeb.route('/modeloDiabetes', methods=['POST'])
def modelo():
    #Procesar datos de entrada 
    contenido = request.json
    print(contenido)
    datosEntrada = np.array([
            contenido['pregnancies'],
            contenido['glucose'],
            contenido['bloodPressure'],
            contenido['skinThickness'],
            contenido['insulin'],
            contenido['BMI'],
            contenido['DPF'],
            contenido['age']
        ])
            
    #Utilizar el modelo
    resultado=dt.predict(datosEntrada.reshape(1,-1))
    #Regresar la salida del modelo
    return jsonify({"Resultado":str(resultado[0])})


@servidorWeb.route('/trainDiabetes', methods=['GET'])
def trainModel():
    dataFrame = pd.DataFrame(requests.get('http://3.80.126.199:8083/diabetes/readRecords').json())
    print(dataFrame.head())
    dataFrame.drop('id',axis=1,inplace=True)
    print(dataFrame.head())
    #Caracteristicas de entrada (Info de los campos del formulario)
    X=dataFrame.drop('class',axis=1)
    #Caracteristicas de salida (Info de los campos del formulario)
    y=dataFrame['class']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    #Modelo
    from sklearn.tree import DecisionTreeClassifier
    dt=DecisionTreeClassifier()
    print("llegue aqui")
    #Entrenar el modelo
    dt.fit(X_train,y_train)
    
    #Evaluación 
    print("Exactitud del modelo", dt.score(X_test,y_test))
    #exportar el modelo creado para usarlo en un servidor web con flask
    dump(dt,'modelo.joblib') #64 bits
    
    return jsonify({"Resultado":dt.score(X_test,y_test)})

    
if __name__ == '__main__':
    servidorWeb.run(debug=False,host='0.0.0.0',port='8080')
