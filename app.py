import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv(), override=False)
API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="LangChainデモ - A/B専門家", page_icon="🍳", layout="centered")

if not API_KEY:
    st.error("OPENAI_API_KEY が読み込めていません。.env もしくは st.secrets を確認してください。")
    st.stop()

def run_llm(user_text: str, expert_key: str) -> str:
    expert_systems = {
        "A": (
            "あなたは和食の料理研究家です。出汁・下味・火入れの要点を丁寧に説明し、"
            "家庭で再現しやすいレシピやコツを、具体的な手順番号つきで提案してください。"
        ),
        "B": (
            "あなたは洋食の料理研究家です。ソテー、デグレーズ、乳化、ソース作りの要点をわかりやすく説明し、"
            "家庭で再現しやすいレシピやコツを、具体的な手順番号つきで提案してください。"
        ),
    }
    system_message = expert_systems.get(expert_key, expert_systems["A"])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{input_text}"),
        ]
    )

    llm = ChatOpenAI(
        api_key=API_KEY,
        model="gpt-4o-mini",
        temperature=0.5 if expert_key == "A" else 0.8,
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser
    result = chain.invoke({"input_text": user_text})
    return result.strip()

st.title("料理専門家に相談するアプリ")

st.markdown(
    """
### アプリの概要
- ラジオボタンで専門家を選び、入力フォームに相談内容を入力してください。
  選択に応じた専門家が回答します。

### 操作方法
1. ラジオボタンで専門家を選択
2. 入力フォームに素材や相談内容を記入
3. 「送信」ボタンを押すと、画面下に回答が表示されます。
"""
)

expert_choice = st.radio(
    "専門家の種類を選んでください：",
    ["和食の料理専門家", "洋食の料理専門家"]
)

user_text = st.text_area(
    "相談内容を入力してください（材料・希望など）",
    height=120,
    placeholder="例：鶏ももと玉ねぎを使って。簡単にできるもの"
)

if st.button("送信", type="primary"):
    if not user_text.strip():
        st.warning("入力テキストを入れてね。")
    else:
        with st.spinner("専門家が考え中…"):
            try:
                answer = run_llm(user_text, expert_choice)
                st.subheader("回答")
                st.markdown(answer)
            except Exception as e:
                st.error("生成時にエラーが発生しました。")
                st.caption(f"{type(e).__name__}: {e}")
