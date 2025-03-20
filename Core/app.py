import os
import re
import gradio
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
# from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from utils.pdf_cleaner import clean_text

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY não encontrada! Verifica o teu .env.")


def split_by_artigo(text):
    """
    Divide o texto sempre que a palavra "Artigo" for encontrada,
    preservando "Artigo" no início de cada chunk.
    """
    chunks = re.split(r'(?=Artigo)', text)
    # Remove espaços em branco e retorna somente chunks não vazios
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def predict(message, history):
    # Phase 1 - Rag Preparation
    # Step 1 - Document Loading
    file_path = "../docs/bitcoin.pdf"

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Step 1.5 - Document cleaning
    for doc in docs:
        # \n ou break lines ou enters, para novos paragrafos é valido e devemos deixar, porque é nova informação
        # o resto poderá ser apagado
        doc.page_content = clean_text(doc.page_content)

    # Step 2 - Document Transform / Splitting
    chunksize = 1000
    chunkoverlap = chunksize * 0.2
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunksize,
        chunk_overlap=chunkoverlap,
        separators=["Artigo"]
    )

    chunks = text_splitter.split_documents(docs)
    # print(chunks)

    # Step 3 - Embedding
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key  # Passando a chave manualmente
    )

    chunks_as_string = [chunk.page_content for chunk in chunks]
    # print(chunks_as_string)
    vectors = embeddings_model.embed_documents(chunks_as_string)

    # print(len(chunks))
    # print(len(vectors))
    # print(len(vectors[0]))

    # vector_db = Chroma(
    #     collection_name="legal-documents",
    #     # Mesma coleção TEM que ter o mesmo embedding model - embedding model de diferente tamanho dá erro
    #     # Mas embedding model com o mesmo tamanho não dá erro (ter atenção aqui)
    #     embedding_function=embeddings_model
    # )

    # Step 4 Store on Vector db
    vector_db = FAISS.from_documents(chunks, embeddings_model)

    # Phase 2 - Rag Generation
    # Step 2.1 - Similarity Search

    relevant_chunks = vector_db.similarity_search(message, k=5)

    print("relevant_chunk")
    print("relevant_chunks:")
    for chunk in relevant_chunks:
        print(chunk)

  # Step 2.2 Criar a nossa PROMPT final

    prompt = f"""
        Instructions: 
        Answer the user query based on the provided context.
        Don't answer questions that are not based on the context.
        If the question is not based on the context, reply with 'I don't have the necessary information to answer that'.

        User Query:
        {message}

        Context:
        {relevant_chunks}
        End of Context
    """

    # Step 2.3 - Chamar a llm da OpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    response = llm.invoke(prompt)

    return response.content


gradio.ChatInterface(predict).launch(debug=True)
