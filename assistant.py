import os 
from dotenv import load_dotenv
from groq import Groq
import chromadb 
from langchain_community.document_loaders import PyPDFLoader            
from langchain_text_splitters import RecursiveCharacterTextSplitter 

load_dotenv()
groq_client = Groq()





#the temp will stay 0.0-0.2 for accuracy 
MODEL_NAME = "openai/gpt-oss-120b"
MAX_TOKEN_LIMIT = 1024
TEMP=0.3


def build_index(pdf_path):
    pages = PyPDFLoader(pdf_path).load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap=200,
    ).split_documents(pages)
    
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("pdf")
    except Exception:
        pass
    collection = client.get_or_create_collection("pdf")
    
    collection.add(
        documents=[c.page_content for c in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"page": c.metadata.get("page", 0)} for c in chunks],
    )
    print(f"Indexed {len(chunks)} chunks.")
    return collection
    
def ask(collection, question):
    results = collection.query(query_texts=[question], n_results=3)
    retrieved = results["documents"][0]


    context = ""

    
    for i, chunk in enumerate(retrieved, start=1):
        context += f"[Source {i}]\n{chunk}\n\n"
    
    system_prompt = (
        "You answer questions using ONLY the provided sources. "
        "Cite the source you used like [Source 1] at the end. "
        "Answers must be TO THE POINT and concise."
        "If the answer is not in the sources, say 'I don't know based on the document.' "
        "Do not use outside knowledge."
    )
    user_prompt = f"Sources:\n{context}\nQuestion: {question}"
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,   
    )
    return response.choices[0].message.content

if __name__ == "__main__":
#pdf path
    collection = build_index("")
    while True:
        q = input("You (type quit to end): ")
        if q.lower() in ("quit"):
            break
        if not q:
            continue
        print("\n" + ask(collection, q))



