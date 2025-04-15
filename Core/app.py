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
        chunk_size=1250,
        chunk_overlap=250,
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


def ingestion_phase():
    file_path = "../docs/codigo_trabalho.pdf"

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    article_chunks = split_by_article(docs)

    # Fallback: Se a divisão por artigos falhar ou retornar poucos chunks, usa split padrão
    if not article_chunks or len(article_chunks) < 5:
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


def resolve_coreference(message, history):
    # Remove pontuação e espaços extras
    cleaned = message.strip().lower()

    # Caso especial: apenas um número ou algo como "e o 137"
    num_only_match = re.match(r"^(e\s+o\s+)?(\d+[a-zA-Z\-]*)\??$", cleaned)
    if num_only_match:
        last_article_ref = None
        for turn in reversed(history):
            if not turn or not turn[0]:
                continue
            match = re.search(
                r'artigo\s+(\d+[.º°A-Z\-]*)', turn[0], re.IGNORECASE)
            if match:
                last_article_ref = match.group(1)
                break

        numero = num_only_match.group(2)
        print(
            f"[coref] Detetado número isolado '{numero}' → reformulado como 'artigo {numero}'")
        return f"artigo {numero}"

    return message


def retrieval_phase(message, history):
    vector_db = Chroma(
        collection_name="legal-documents",
        embedding_function=embeddings_model,
        persist_directory="./db_v3",
    )

    message = resolve_coreference(message, history)

    artigo_match = re.search(
        r'artigo(?:\s+n[.º°]?\s*|\s+)?(\d+[.º°A-Z\-]*)', message, re.IGNORECASE)

    matched_chunks = []

    if artigo_match:
        numero_artigo = artigo_match.group(1).strip()
        numero_artigo = numero_artigo.replace(
            "º", "").replace("°", "").replace(".", "").lower()

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

    return matched_chunks


def generation_phase(message, history):
    matched_chunks = retrieval_phase(message, history)

    context_str = "\n\n".join([
        f"{chunk.metadata.get('article_title', '')}\n{chunk.page_content.strip()}"
        for chunk in matched_chunks
    ])

    cited_articles = list({
        chunk.metadata.get("article_title")
        for chunk in matched_chunks
        if chunk.metadata.get("article_title")
    })

    artigo_info = (
        f"Baseado nos seguintes artigos: {', '.join(cited_articles)}.\n\n"
        if cited_articles else ""
    )

    formatted_history = "\n".join(
        f"Pergunta do utilizador: {turn[0]}\nResposta do assistente: {turn[1]}"
        for turn in history if len(turn) == 2
    )

    prompt = f"""
    Instruções:
    Se for algo conversacional, responde como um assistente de chat.
    Tenta sempre responder em português europeu.
    Responde à pergunta do utilizador com base no contexto e/ou no histórico abaixo.
    Usa tanto o contexto como o histórico da conversa para responder ao utilizador.
    Se a resposta não estiver no contexto e/ou no histórico, responde apenas com: 'Não tenho informação suficiente para responder a isso.'.

    Quando a resposta estiver no contexto e/ou no histórico, refere os artigos usados com base no título (ex: Artigo 114.º), se estiverem disponíveis.
    Não dês o número do artigo se não tiveres a certeza de que é o correto.

    Pergunta:
    {message}
    Fim da pergunta

    {artigo_info}

    Contexto:
    {context_str}
    Fim do contexto
    
    Histórico:
    {formatted_history}
    Final do histórico.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    response = llm.invoke(prompt)
    return response.content


ingestion_phase()
gradio.ChatInterface(generation_phase).launch(debug=True)
