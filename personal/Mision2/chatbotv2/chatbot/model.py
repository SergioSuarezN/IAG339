import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

MODEL_DIR="models"
MOLDEL_PATH=os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH=os.path.join(MODEL_DIR, "vectorizer.pkl")
ANSWER_PATH=os.path.join(MODEL_DIR, "vectorizer.pkl")

def buid_and_train_model(train_pairs):
    questions = [q for q, _ in train_pairs]
    answers = [a for _, a in train_pairs]
    vectorizer = CountVectorizer()
    x=vectorizer.fit_transform(questions)

    unique_answers = sorted(set(answers))
    answer_to_label= {a:i for i,a in enumerate(unique_answers)}
    y = [answer_to_label[a] for a in answers]
    model = MultinomialNB()
    model.fit(x,y)
    # Crear carpeta para el modelo si no existe
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Guardar los objetos entrenados
    with open(MOLDEL_PATH, "wb") as f:
        pickle.dump(model,f)
    with open(VECTORIZER_PATH,"wb") as f:
        pickle.dump(vectorizer,f)
    with open(ANSWER_PATH, "wb") as f:
        pickle.dump(unique_answers,f)
    print("Modelo entrenado y guardado correctamente.")
    return model, vectorizer, unique_answers

def load_model():
    if(
        os.path.exists(MODEL_DIR)
        and os.path.exists(VECTORIZER_PATH)
        and os.path.exists(ANSWER_PATH)
    ):
        with open(MOLDEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer= pickle.load(f)
        with open(ANSWER_PATH, "rb") as f:
            unique_answers = pickle.load(f)
        print("Modelo cargado desde disco")
    else :
        print("No hay modelo guardado, sera necesario entrenarlo")
        return None, None, None
    
def predict_answer(model, vectorizer, unique_answers,user_text):
    x = vectorizer.transform([user_text])
    label = model.predict(x)[0]
    return unique_answers[label]