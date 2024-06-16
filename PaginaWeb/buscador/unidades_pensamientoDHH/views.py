from django.shortcuts import render
from django.db import connection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def buscar_unidades_pensamientoDHH(parrafo):
    with connection.cursor() as cursor:
        cursor.execute("SELECT contenido, libro, capitulo, versiculos FROM unidades_pensamiento")
        resultados = cursor.fetchall()
    
    vectorizador = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        smooth_idf=False
    )

    contenidos = [resultado[0] for resultado in resultados]
    referencias = [(resultado[1], resultado[2], resultado[3]) for resultado in resultados]

    tfidf_matrix = vectorizador.fit_transform(contenidos)
    parrafo_tfidf = vectorizador.transform([parrafo])
    similitudes = cosine_similarity(parrafo_tfidf, tfidf_matrix)

    unidades_similares = []
    for similitud in sorted(similitudes[0], reverse=True):
        if similitud >= 0.05:
            indice_similitud = list(similitudes[0]).index(similitud)
            unidad_pensamiento_similar = contenidos[indice_similitud]
            referencia_similar = referencias[indice_similitud]
            unidades_similares.append({
                'similitud': similitud,
                'contenido': unidad_pensamiento_similar,
                'libro': referencia_similar[0],
                'capitulo': referencia_similar[1],
                'versiculo': referencia_similar[2]
            })

    return unidades_similares

def index(request):
    return render(request, 'unidades_pensamientoDHH/index.html')

def resultados(request):
    if request.method == 'GET' and 'q' in request.GET:
        query = request.GET.get('q', '').strip()
        resultados = buscar_unidades_pensamientoDHH(query)
        return render(request,  'unidades_pensamientoDHH/resultados.html', {'resultados': resultados, 'query': query})
    return render(request, 'unidades_pensamientoDHH/index.html')
