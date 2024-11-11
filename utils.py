import os
import pandas as pd
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import OpenAIEmbeddings
from kiwipiepy.utils import Stopwords
from kiwipiepy import Kiwi
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

df = pd.read_excel("data/경희대학교_음식데이터_1108.xlsx")
# kiwi 지정
kiwi = Kiwi(typos="basic", model_type="sbg")
stopwords = Stopwords()
stopwords.remove(("사람", "NNG"))

# Embedding
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", api_key=os.environ["OPENAI_API_KEY"]
)


def kiwi_tokenize(text):
    text = "".join(text)
    result = kiwi.tokenize(text, stopwords=stopwords, normalize_coda=True)
    N_list = [i.form.lower() for i in result if i.tag in ["NNG", "NNP", "SL"]]
    return N_list


def retrieve_text():
    message = (
        df["가게"] + df["메뉴"] + df["위치"] + df["전화번호"] + df["URL"]
    ).values.tolist()
    kiwi_bm25 = BM25Retriever.from_texts(message, preprocess_func=kiwi_tokenize)
    kiwi_bm25.k = 2
    faiss = FAISS.from_texts(message, embeddings).as_retriever(search_kwargs={"k": 2})
    kiwibm25_faiss_46 = EnsembleRetriever(
        retrievers=[kiwi_bm25, faiss], weights=[0.4, 0.6], search_type="mmr"
    )
    return kiwibm25_faiss_46
