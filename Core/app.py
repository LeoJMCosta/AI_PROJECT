import os
import re
import chromadb.api.client
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

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY não encontrada! Verifica o teu .env.")

vector_db = None
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai_api_key
)


def split_by_article(documents):
    full_text = " ".join([doc.page_content for doc in documents])
    metadata = documents[0].metadata if documents else {}

    pattern = re.compile(r'(Artigo\s+\d+[.ºA-Z\-]*)')
    matches = list(pattern.finditer(full_text))

    article_chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + \
            1 < len(matches) else len(full_text)
        chunk_text = full_text[start:end].strip()
        article_chunks.append(
            Document(page_content=chunk_text, metadata=metadata))

    return article_chunks


def load_information():
    # Phase 1 - Rag Preparation
    # Step 1 - Document Loading
    file_path = "../docs/codigo_trabalho.pdf"

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Step 1.5 - Document cleaning
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    chunks = split_by_article(docs)

    chunks_as_string = [chunk.page_content for chunk in chunks]
    vectors = embeddings_model.embed_documents(chunks_as_string)

    chromadb.api.client.SharedSystemClient.clear_system_cache()
    vector_db = Chroma(
        collection_name="legal-documents",
        # Mesma coleção TEM que ter o mesmo embedding model - embedding model de diferente tamanho dá erro
        # Mas embedding model com o mesmo tamanho não dá erro (ter atenção aqui)
        embedding_function=embeddings_model,
        persist_directory="./db_v2",
    )

    vector_db.add_documents(chunks)


def generation_phase(message, history):

    vector_db = Chroma(
        collection_name="legal-documents",
        # Mesma coleção TEM que ter o mesmo embedding model - embedding model de diferente tamanho dá erro
        # Mas embedding model com o mesmo tamanho não dá erro (ter atenção aqui)
        embedding_function=embeddings_model,
        persist_directory="./db_v2",
    )
    relevant_chunks = vector_db.similarity_search(message, k=1)

    print("relevant_chunks:")
    for chunk in relevant_chunks:
        print(chunk)

    context_str = "\n\n".join(
        [chunk.page_content for chunk in relevant_chunks])

  # Step 2.2 Criar a nossa PROMPT final

    prompt = f"""
        Instructions
        Answer ther user query based on the provided context, or use the provided context and logically answer the user question.
        Don't asnwer questions that are not present on the provided context or you can't logically answer it from the context.
        If the question is not on the provided context, answer with 'Não tenho informação suficiente para responder a isso.'.

        The source on the metadata is about the file name, use it on your answer to say what file did you used.

        User Query
        {message}
        End of User Query

        Context
        {context_str}
        End of Context
    """

    # Step 2.3 - Chamar a llm da OpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    response = llm.invoke(prompt)

    print("response")

    return response.content


load_information()
gradio.ChatInterface(generation_phase).launch(debug=True)
