import nltk

try:
    nltk.download('punkt')
    print(" NLTK punkt descargado correctamente")
except Exception as e:
    print("Error durante la descarga:",e)
