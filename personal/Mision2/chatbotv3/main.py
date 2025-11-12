from chatbot.data import training_data
from chatbot.model import buid_and_train_model, predict_answer, load_model

def chat(model, vectorizer, unique_answers):
    """Inicia el modelo de conversacion"""
    print("\n 📨 Chat Inicado. Escribe salir para terminar \n")
    while True:
        user = input("Tu: ").strip()
        if user.lower() in {"salir","exit","quit"}:
            print("Bot: ¡Hasta pronto!")
            break
        response = predict_answer(model, vectorizer, unique_answers,user)
        print("Bot:", response)
        
        
        
def main():
    # Intentar cargar el modelo
    model, vectorizer, unique_answers=load_model()
    # Menu principal
    while True:
        print("\n=== 🤖 MENU PRINCIPAL DEL CHATBOT===")
        print("1️⃣ Chatear con el modelo")
        print("2️⃣ Reentrenar el modelo.")
        print("3️⃣ Salir.")
        opcion = input("\n Elige una opcion (1-3): ").strip()
        if opcion == "1":
            if model is None:
                print("\n ⚠️ No hay modelo entrenado. Entrenalo primero.")
            else:
                chat( model, vectorizer, unique_answers)
        elif opcion == "2":
            print("\n 🔂 Reentrenando el modelo con los nuevos datos...")
            model, vectorizer, unique_answers = buid_and_train_model(training_data)
            print("🆗 Modelo Actualizado correctamente.")
        elif opcion == "3":
            print("\n 👋 !Hasta Luego!")
            break
        else :
            print("\n ❌⛔ Opcion no valida. Intente nuevamente!!")
        
if __name__=="__main__":
    main()