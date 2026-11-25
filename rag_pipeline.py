from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
#from langchain_classic.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate


file_path = "./data/ODI_rules.pdf"
def load_file(file_path:str):
    loader=PyPDFLoader(file_path=file_path)
    docs = loader.load()
    return docs

def chunking(docs, chunk_size = 800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return chunks

def get_embeddings():
    embeddings = OpenAIEmbeddings()
    return embeddings

def create_vector_db(chunks,embeddings):
    vector_db = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    location=":memory:",
    collection_name="rag_docs",
)
    return vector_db

def get_hybrid_retriver(vector_db, chunks, top_k:int=10):
    vector_retriver = vector_db.as_retriever(search_type = "mmr", search_kwargs={"k" : top_k, "fetch_k" : 30 })

    bm25retriver = BM25Retriever.from_documents(documents=chunks)
    bm25retriver.k = top_k

    retriver = EnsembleRetriever(retrievers=[vector_retriver, bm25retriver], weights=[0.6,0.4])

    return retriver


def add_reranker(retriver, top_n:int=4):
    cohere = CohereRerank(model="rerank-v3.5", top_n=top_n)
    reranker = ContextualCompressionRetriever(base_retriever=retriver, base_compressor=cohere)
    return reranker


def get_llm():
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm

def get_prompt():
    prompt = PromptTemplate.from_template(
        """
    Act as helpful assistant and Give the correct answer of the Question from context. If context is insufficient then give answer as 'Not Found'.
    context :\n {context} \n\n
    question : {query}
"""
    )
    return prompt

def create_chain(prompt,llm):
    chain = prompt|llm
    return chain


def ask_query(query, retriver, chain):
    docs = retriver.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    response = chain.invoke({"context" : context, "query":query})
    return response.content


def rag_pipeline(file_path, query):
    docs = load_file(file_path=file_path)
    chunks = chunking(docs)
    embeddings = get_embeddings()
    db = create_vector_db(chunks, embeddings)
    retriver = get_hybrid_retriver(db,chunks)
    reranked_retriever = add_reranker(retriver)
    llm = get_llm()
    prompt = get_prompt()
    chain = create_chain(prompt,llm)

    result = ask_query(query,reranked_retriever, chain)
    return result






