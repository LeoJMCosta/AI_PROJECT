import re
import gradio
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_text_splitters import CharacterTextSplitter


def predict(message, history):
    # Phase 1
    file_path = "../docs/codigo_trabalho.pdf"

    loader = PyPDFLoader(file_path, mode="single")
    docs = loader.load()

    # Step 1.5 - Document cleaning
    for doc in docs:
        # \n ou break lines ou enters, para novos paragrafos é valido e devemos deixar, porque é nova informação
        # o resto poderá ser apagado
        doc.page_content = re.sub('\n', '', doc.page_content)

    print(docs)

    # Step 2 - Document Transfrom / Splitting
    chunksize = 500
    chunkoverlap = chunksize * 0.2
    text_splitter = RecursiveCharacterTextSplitter(
        # Este só se usa no CharacterTextSplitter
        # separator= '. \n',
        chunk_size=chunksize,
        chunk_overlap=chunkoverlap
    )

    chunks = text_splitter.split_documents(docs)
    print(chunks)

    return "olá!"


gradio.ChatInterface(predict).launch(debug=True)
