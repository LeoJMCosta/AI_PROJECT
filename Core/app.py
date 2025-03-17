import gradio
from langchain_community.document_loaders import PyPDFLoader


def predict(message, history):
    # Phase 1
    file_path = "../docs/codigo_trabalho.pdf"

    loader = PyPDFLoader(file_path, mode="single")
    docs = loader.load()

    print(docs)

    return "olá!"


gradio.ChatInterface(predict).launch(debug=True)
