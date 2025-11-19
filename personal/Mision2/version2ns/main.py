# main.py
from flask import Flask, render_template, request, jsonify
from chatbot.data import training_data
from chatbot.model import build_and_train_model, load_model, predict_cluster
import random 

app = Flask(__name__)

# Intentamos cargar el modelo (o entrenamos si no existe)
model, vectorizer = load_model()
if model is None:
    model, vectorizer = build_and_train_model(training_data, n_clusters=6)  # ✅ Número de grupos ajustable


#Respuesta por grupo
RESPUESTAS = {
    0:[
        "!Hola! 😀 ¿ Cómo estás ?",
       "!Qué gusto saludarte!",
       "!Hola! ¿ En que puedo ayudarte?",
       ],
    1:[
        "Hasta luego 👋!",
        "Nos vemos pronto. ",
        "Cuidate. Espero verte de nuevo",
        ],
    2:[
        "Soy un asistente virtual creado para ayudarte 🤖",
        "!Por supuesto! ¿ Con qué necesitas ayuda ?",
        "Cuéntame tu problema y buscaré una solución",
       ],
    3:[
        "Puedo ofrecerte información o resolver tus dudas",
        "! En que puedo ayudar",
        "Estoy aquí para resolver tus preguntas",
        ],
    4:[
        "! Gracias a ti ! 🤗",
        "De nada, me alegra ser de ayuda",
        "!Muy amable de tu parte!",
        ],
    5:[
        "Lamento que te sientas así, puedo intentarlo de nuevo",
        "Parece que algo no salió bien, ¿ Quieres que lo revisemos",
        "No siempre soy perfecto, pero puedo intentarlo otra vez.",
        ]
}
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_text = request.form.get("message", "")
    if not user_text.strip():
        return jsonify({"response": "Por favor escribe algo 😅"})

    # Predice el grupo al que pertenece el mensaje
    cluster = predict_cluster(model, vectorizer, user_text)

    # ✅ Mensaje más descriptivo
    #response = f"Tu mensaje pertenece al grupo {cluster}. Este grupo contiene frases con significados similares."
    response = random.choice(RESPUESTAS.get(cluster, [
        "No estoy seguro de emtender, pero puedo intentarlo otra vez."
    ]))


    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
