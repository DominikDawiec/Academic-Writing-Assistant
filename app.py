import streamlit as st
import openai
import pinecone
import os

# Add the UI title, text input and search button
st.title("Q&A App 🔎")
query = st.text_input("Zadaj pytanie")

if st.button("Szukaj"):
   
    # Get Pinecone API environment variables
    pinecone_api = ' '
    pinecone_env = ' '
    pinecone_index = ' '
    
    # Get OpenAI API Key
    openai.api_key = ' '
    gpt4_model_name = "gpt-3.5-turbo"
 
    # Initialize Pinecone client and set index
    pinecone.init(api_key=pinecone_api, environment=pinecone_env)
    index = pinecone.Index(pinecone_index)
 
    # Convert your query into a vector using OpenAI
    try:
        query_vector = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error calling OpenAI Embedding API: {e}")
        st.stop()
 
    # Search for the 3 most similar vectors in Pinecone
    search_response = index.query(
        top_k=1,
        vector=query_vector,
        include_metadata=True)
        
    # Combine the 3 vectors into a single text variable that it will be added in the prompt
    chunks = search_response['matches']

    # Write which are the selected chunks in the UI for debugging purposes
    with st.expander("Chunks"):
        for i, chunk in enumerate(chunks):
            text = chunk["metadata"]['text'].replace("\n", " ")
            source = chunk["metadata"]['source']
            score = chunk['score']
            st.write(f"Tekst: {text}\n")
            st.write(f"Źróło: {source}\n")
            #st.write(f"Dopasowanie: {score:.2f}\n")
    
    with st.spinner("Przygotowywanie odpowiedzi..."):
        try:
            # Build the prompt
            prompt = f"""
            Odpowiedz na poniższe pytanie, opierając się na podanym kontekście. 
            Odpowiedź powinna być w stylu charakterystycznym dla prac dyplomowych. Powinna mieć tylko jeden akapit.
            ---
            QUESTION: {query}                                            
            ---
            CONTEXT:
            {text}
            """
             
            # Run chat completion using GPT-4
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    { "role": "system", "content":  "Jesteś asystentem w pisaniu prac dyplomowych." },
                    { "role": "user", "content": prompt }
                ],
                temperature=0.2,
                max_tokens=800
            )
 
            # Get the response from GPT-4
            st.markdown("### Odpowiedź:")
            st.write(response.choices[0]['message']['content'])
   
   
        except Exception as e:
            st.error(f"Error with OpenAI Chat Completion: {e}")