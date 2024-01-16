import streamlit as st
import openai
import pinecone
import os

# Dodanie tytułu UI, pola do wpisywania tekstu i przycisku wyszukiwania
st.title("Q&A App 🔎")
query = st.text_input("Zadaj pytanie")

if st.button("Szukaj"):
    # Wczytywanie zmiennych środowiskowych
    pinecone_api = '1a54cacc-bc0c-415e-bab6-e91a7a69f01a'
    pinecone_env = 'gcp-starter'
    pinecone_index = 'knad'
    openai_api_key = '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'

    # Sprawdzenie, czy wszystkie zmienne środowiskowe zostały ustawione
    if not all([pinecone_api, pinecone_env, pinecone_index, openai_api_key]):
        st.error("Brakujące zmienne środowiskowe. Proszę je prawidłowo ustawić.")
        st.stop()

    # Inicjalizacja klienta Pinecone i ustawienie indeksu
    pinecone.init(api_key=pinecone_api, environment=pinecone_env)
    index = pinecone.Index(pinecone_index)

    # Konwersja zapytania na wektor za pomocą OpenAI
    try:
        query_vector = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002",
            api_key=openai_api_key
        )["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Błąd podczas wywoływania API OpenAI Embedding: {e}")
        st.stop()

    # Wyszukiwanie podobnych wektorów w Pinecone
    try:
        search_response = index.query(
        top_k=1,  
        vector=query_vector,
        include_metadata=True
    )
    except Exception as e:
        st.error(f"Błąd podczas wyszukiwania w Pinecone: {e}")
        st.stop()

    # Łączenie wektorów w jedną zmienną tekstową
    chunks = [chunk["metadata"] for chunk in search_response['matches']]

    # Wyświetlanie wybranych fragmentów w UI dla celów debugowania
    with st.expander("Chunks"):
        for chunk in chunks:
            text = chunk['text'].replace("\n", " ")
            source = chunk['source']
            st.write(f"Tekst: {text}\n\n Źródło: {source}\n")

    # Przygotowanie odpowiedzi z użyciem GPT-4
    with st.spinner("Przygotowywanie odpowiedzi..."):
        try:
            # Budowanie promptu
            prompt = f"""
            Odpowiedz na poniższe pytanie, opierając się na podanym kontekście. 
            Odpowiedź powinna być w stylu charakterystycznym dla prac dyplomowych, z jednym akapitem.
            ---
            PYTANIE: {query}                                            
            ---
            KONTEKST:
            """
            for chunk in chunks:
                prompt += f"{chunk['text']}\n---\n"
                        
            # Uruchomienie kompletacji czatu z wykorzystaniem GPT-3.5
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Jesteś asystentem w pisaniu prac dyplomowych."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800,
                api_key=openai_api_key
            )

            # Pobranie odpowiedzi od GPT-3.5
            st.markdown("### Odpowiedź:")
            st.write(response.choices[0]['message']['content'])
        except Exception as e:
            st.error(f"Błąd podczas korzystania z OpenAI Chat Completion: {e}")
