import os
import pickle
from flask import Flask, jsonify, request
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression


colunas = ["LotArea", "YearBuilt", "GarageCars"]
model = pickle.load(open("models/modelo.sav", "rb"))

# Initialize web app
app = Flask(__name__)
app.config["BASIC_AUTH_USERNAME"] = os.environ.get("BASIC_AUTH_USERNAME")
app.config["BASIC_AUTH_PASSWORD"] = os.environ.get("BASIC_AUTH_PASSWORD")
print("BASIC_AUTH_USERNAME = " + os.environ.get("BASIC_AUTH_USERNAME"))
print("BASIC_AUTH_PASSWORD = " + os.environ.get("BASIC_AUTH_PASSWORD"))

basic_auth = BasicAuth(app)

@app.route("/")
def home():
    return "Minha primeira API."

@app.route("/sentimento/<frase>")
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang="pt", to="en")
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)

@app.route("/cotacao/", methods=["POST"])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = model.predict([dados_input])
    return jsonify(preco=preco[0])

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")