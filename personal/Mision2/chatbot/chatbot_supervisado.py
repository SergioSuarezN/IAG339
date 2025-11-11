from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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
    return model, vectorizer, unique_answers
def predict_answer(model, vectorizer, unique_answers,user_text):
    x = vectorizer.transform([user_text])
    label = model.predict(x)[0]
    return unique_answers[label]
if __name__ == "__main__":
    training_data =[
        ("hola","!Hola ¿En qué puedo ayudarte?"),
        ("buenos días","!Buenos Días¡"),
        ("cómo estás","Estoy bien, gracias por preguntar"),
        ("adiós","!Hasta luego¡"),
        ("tu nombre","Soy un chatbot de ejemplo"),
        ("que puedes hacer"," Puedo responder preguntas simples basadas en ejemplos"),
    ]
    model, vectorizer, unique_answers= buid_and_train_model(training_data)
    print("Chatbot supervisadi listo, Escribe 'salir' para terminar")

    while True:
        user = input("Tu: ").strip()
        if user.lower() in {"salir","exit","quit"}:
            print("Bot: ¡Hasta pronto!")
            break
        response = predict_answer(model, vectorizer, unique_answers,user)
        print("Bot:", response)
    
