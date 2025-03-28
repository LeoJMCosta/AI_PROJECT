import os
import re
import gradio
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import chromadb.api
from utils.pdf_cleaner import clean_text
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

vector_db = None
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai_api_key
)


def split_by_article(documents):
    full_text = " ".join([doc.page_content for doc in documents])
    metadata = documents[0].metadata if documents else {}

    pattern = re.compile(
        r'(Artigo\s+(\d+[.ºA-Z\-]*))\s+(.*?)(?=Artigo\s+\d+[.ºA-Z\-]*|\Z)', re.DOTALL)
    article_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    for match in pattern.finditer(full_text):
        full_title = match.group(1).strip()
        number_only = match.group(2).strip()
        body = match.group(3).strip()
        content = f"{full_title}\n{body}"

        chunk_metadata = {
            **metadata,
            "article_number": number_only,
            "article_title": full_title
        }

        subchunks = splitter.split_text(content)
        for sub in subchunks:
            article_chunks.append(
                Document(page_content=sub, metadata=chunk_metadata)
            )

    return article_chunks


def load_information():
    file_path = "../docs/codigo_trabalho.pdf"

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    # Tenta dividir por artigos primeiro
    article_chunks = split_by_article(docs)

    # Fallback: Se a divisão por artigos falhar ou retornar poucos chunks, usa split padrão
    if not article_chunks or len(article_chunks) < 10:
        print("Fallback: usando RecursiveCharacterTextSplitter.")
        chunksize = 1250
        chunkoverlap = chunksize * 0.2
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunksize,
            chunk_overlap=chunkoverlap
        )
        chunks = text_splitter.split_documents(docs)
    else:
        chunks = article_chunks

    chromadb.api.client.SharedSystemClient.clear_system_cache()
    vector_db = Chroma(
        collection_name="legal-documents",
        embedding_function=embeddings_model,
        persist_directory="./db_v3",
    )
    vector_db.add_documents(chunks)


def generation_phase(message, history):
    vector_db = Chroma(
        collection_name="legal-documents",
        embedding_function=embeddings_model,
        persist_directory="./db_v3",
    )

    artigo_match = re.search(
        r'artigo(?:\s+n[.ºº°]?\s*|\s+)?(\d+[.º°A-Z\-]*)', message, re.IGNORECASE)

    matched_chunks = []

    if artigo_match:
        numero_artigo = artigo_match.group(1).strip()
        numero_artigo = numero_artigo.replace(
            "º", "").replace("°", "").replace(".", "").lower()

        print(
            f"🔎 A tentar encontrar diretamente o artigo {numero_artigo} nos metadados...")

        all_docs = vector_db.get()
        for i, content in enumerate(all_docs["documents"]):
            metadata = all_docs["metadatas"][i]
            artigo_meta = metadata.get("article_number", "").strip()
            artigo_meta = artigo_meta.replace("º", "").replace(
                "°", "").replace(".", "").lower()

            if artigo_meta == numero_artigo:
                matched_chunks.append(
                    Document(page_content=content, metadata=metadata)
                )

    if not matched_chunks:
        query = f"De acordo com o Código do Trabalho português, {message}"
        matched_chunks = vector_db.similarity_search(query, k=4)

    print("relevant_chunks:")
    for chunk in matched_chunks:
        print(chunk.metadata)

    context_str = "\n\n".join([chunk.page_content for chunk in matched_chunks])

    cited_articles = list({
        chunk.metadata.get("article_title")
        for chunk in matched_chunks
        if chunk.metadata.get("article_title")
    })

    artigo_info = (
        f"Baseado nos seguintes artigos: {', '.join(cited_articles)}.\n\n"
        if cited_articles else ""
    )

    prompt = f"""
    Instruções:
    Responde à pergunta do utilizador com base no contexto abaixo.
    Se a resposta não estiver no contexto, responde apenas com: 'Não tenho informação suficiente para responder a isso.'.

    Quando a resposta estiver no contexto, refere os artigos usados com base no título (ex: Artigo 114.º), se estiverem disponíveis.

    Pergunta:
    {message}

    {artigo_info}

    Contexto:
    {context_str}
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    response = llm.invoke(prompt)
    return response.content


load_information()
gradio.ChatInterface(generation_phase).launch(debug=True)
