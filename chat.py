import streamlit as st


from dotenv import load_dotenv


from llm import get_ai_response

st.set_page_config(page_title="GigE 카메라 챗봇", page_icon="📷")

st.title("📷 GigE 카메라 챗봇")
st.caption("GigE/10GigE 카메라 매뉴얼에 기반해 답변해드립니다.")

load_dotenv()

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])




if user_question := st.chat_input(placeholder="GigE/10GigE 카메라에 대해 궁금한 내용을 질문해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("매뉴얼을 검토하는 중입니다"):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
