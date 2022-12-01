from flask import Flask, request, jsonify, render_template
import numpy as np
from joblib import load
from werkzeug.utils import secure_filename
import os
#Extracción de datos
import pandas as pd
#Biblioteca de manejo de vectores y matrices
import numpy as np
#Biblioteca para exportar procesos de machine learning
from joblib import load, dump

#Cargar el modelo
dt = load('modelo.joblib')

#Generar el servidor (Back-end)
servidorWeb = Flask(__name__)


@servidorWeb.route("/formulario",methods=['GET'])
def formulario():
    return render_template('pagina.html')

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
@servidorWeb.route('/modeloForm', methods=['POST'])
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
@servidorWeb.route('/modelo', methods=['POST'])
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


@servidorWeb.route('/retrainModel', methods=['GET'])
def retrainModel():
    # Obtener todos los registros de la base de datos

    #Aqui se reentrena el modelo
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

    # Utilizar el modelo
    resultado = dt.predict(datosEntrada.reshape(1, -1))
    # Regresar la salida del modelo
    return jsonify({"Resultado": str(resultado[0])})


if __name__ == '__main__':
    servidorWeb.run(debug=False,host='0.0.0.0',port='8080')
