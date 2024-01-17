import streamlit as st
import openai
import pinecone
import os

# Prosty front end
st.title("Aplikacja Q&A 🔎")
query = st.text_input("Zadaj pytanie")

if st.button("Szukaj", use_container_width = True):
   
    # Pobierz zmienne środowiskowe Pinecone API
    pinecone_api = ''
    pinecone_env = ''
    pinecone_index = ''
    
    # Pobierz klucz API OpenAI
    openai.api_key = ''
    gpt4_model_name = "gpt-3.5-turbo"
 
    # Zainicjuj klienta Pinecone
    pinecone.init(api_key=pinecone_api, environment=pinecone_env)
    index = pinecone.Index(pinecone_index)
 
    # Konwertuj swoje zapytanie na wektor przy użyciu OpenAI
    try:
        query_vector = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Błąd przy wywoływaniu API OpenAI Embedding: {e}")
        st.stop()
 
    # Wyszukaj 1 najbardziej podobny wektor w Pinecone
    search_response = index.query(
        top_k=1,
        vector=query_vector,
        include_metadata=True)
        
    # Gdy wybierzesz więcej wektorów to zostaną połączone
    chunks = search_response['matches']

    # Wyświetl wybrane dane
    with st.expander("Fragmenty"):
        for i, chunk in enumerate(chunks):
            text = chunk["metadata"]['text'].replace("\n", " ")
            source = chunk["metadata"]['source']
            score = chunk['score']
            st.write(f"Tekst: {text}\n")
            st.write(f"Źródło: {source}\n")
            #st.write(f"Dopasowanie: {score:.2f}\n")
    
    with st.spinner("Przygotowywanie odpowiedzi..."):
        try:
            # Zbuduj monit
            prompt = f"""
            Opisz podany koncept, opierając się na podanym kontekście. 
            Odpowiedź powinna być w stylu charakterystycznym dla prac dyplomowych. Powinna mieć tylko jeden akapit.

            KONCEPT: {query}                                            
            KONTEKST: {text}
            """
             
            # Uruchom uzupełnianie rozmowy przy użyciu GPT-3.5
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    { "role": "system", "content":  "Jesteś asystentem w pisaniu prac dyplomowych." },
                    { "role": "user", "content": prompt }
                ],
                temperature=0.2,
                max_tokens=800
            )
 
            # Pobierz odpowiedź od GPT-3.5
            st.markdown("### Odpowiedź:")
            st.write(response.choices[0]['message']['content'])
   
   
        except Exception as e:
            st.error(f"Błąd przy wywoływaniu OpenAI Chat Completion: {e}")
