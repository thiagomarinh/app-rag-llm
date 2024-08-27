from operator import itemgetter
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker

from langserve.pydantic_v1 import BaseModel, Field


# carregando o documento pdf e dividindo 
def load_split():
    loader = PDFPlumberLoader("podcast.pdf")
    docs = loader.load()
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)
    return documents

_TEMPLATE = """Dada a conversa a seguir e uma pergunta de acompanhamento, reformule a frase 
pergunta de acompanhamento para ser uma pergunta independente, em português.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

# Recuperação do vectorDataBase
def retriever_vector_db():
    vectorstore = FAISS.from_documents(load_split(), embedding=HuggingFaceEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever

llm = Ollama(model="llama3")

# criando uma chain para usar como entrada para a LLM
_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever_vector_db() | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


#Entrada do usuario
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str


conversational_qa_chain = (
    _inputs | _context | ANSWER_PROMPT | llm | StrOutputParser()
)
chain = conversational_qa_chain.with_types(input_type=ChatHistory)
