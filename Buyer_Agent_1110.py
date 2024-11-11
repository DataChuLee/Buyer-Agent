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
from langchain_core.messages.chat import ChatMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import retrieve_text, kiwi_tokenize
from prompt import user_History_prompt_new, prompt_new
from operator import itemgetter
import streamlit as st
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# env 파일에서 OPENAI API KEY 들여옴
load_dotenv()

# LangChain 추적 시작
logging.langsmith("1108_Test_BuyerAgent")

# kiwi 지정
kiwi = Kiwi(typos="basic", model_type="sbg")
stopwords = Stopwords()
stopwords.remove(("사람", "NNG"))

# Embedding
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", api_key=os.environ["OPENAI_API_KEY"]
)

# LLM 설정
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

# 사용자 History 특징 chain
chain_user_history = user_History_prompt_new | llm | StrOutputParser()

# 사용자_History 데이터
df_user = pd.read_excel("data/사용자_History_데이터_1110.xlsx")

# 판매자 데이터
df = pd.read_excel("data/경희대학교_음식데이터_1108.xlsx")

st.title("Buyer Agent")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# Chain 저장용
if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

# 대화 내용을 기억하기 위한 저장소 생성
if "store" not in st.session_state:
    st.session_state["store"] = {}

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# Chain 생성
def create_chain():
    # Prompt 생성
    prompt = prompt_new

    # LLM 생성
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

    # retriever 생성
    retriever = retrieve_text()

    # Chain 생성
    chain = (
        {
            "question": itemgetter("question"),
            "user_history": lambda _: chain_user_history.invoke({"data": df_user}),
            "context": lambda _: itemgetter("question") | retriever,
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지 키
    )

    return rag_with_history


# 초기화 버튼일 눌리면..
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("구매하고 싶은 음식을 입력해주세요")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

if st.session_state["chain"] is None:
    st.session_state["chain"] = create_chain()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        config = {"configurable": {"session_id": "abc123"}}
        response = chain.stream({"question": user_input}, config=config)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
                # Agent와 Message를 줄바꿈하여 출력
                formatted_answer = (
                    ai_answer.replace("음식 및 판매처 정보", "\n음식 및 판매처 정보")
                    .replace("[구매방식]", "\n[구매방식]")
                    .replace("오프라인:", "\n오프라인:")
                    .replace("온라인:", "\n온라인:")
                    .replace("판매처:", "\n판매처:")
                    .replace("메뉴:", "\n메뉴:")
                    .replace("위치:", "\n위치:")
                    .replace("연락처:", "\n연락처:")
                    .replace("구매링크:", "\n구매링크:")
                    .replace("추천 이유:", "\n추천 이유:")
                )
                container.markdown(formatted_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
