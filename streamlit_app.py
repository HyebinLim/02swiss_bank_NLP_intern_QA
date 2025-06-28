import streamlit as st
from helper import get_openai_api_key
from utils import get_doc_tools
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

# --- 앱 설명 표시 ---
st.header("🏦Want a Swiss Bank AI/NLP job? Ask me!")
st.markdown("""
- I found AI/NLP scientist position at a bank in Zurich, Switzerland on the last year of my master's at University of Zurich.
- I documented my full preparation journey on my blog [(89% 직장인 일지)](https://blog.naver.com/imyourbest)
- A lot of people have asked me for tips — so I launched a little Q&A bot to answer your questions!
""")

# --- API 키 입력 ---
if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state['OPENAI_API_KEY'] = ''

api_key = st.text_input(
    "🔑 Enter your OpenAI API key:",
    type="password",
    value=st.session_state['OPENAI_API_KEY']
)

if api_key:
    st.session_state['OPENAI_API_KEY'] = api_key

# --- Q&A 기능 준비 ---
if st.session_state['OPENAI_API_KEY']:
    # 문서 도구 및 에이전트 준비 (캐싱)
    @st.cache_resource(show_spinner=True)
    def load_tools_and_agent(api_key):
        import os
        os.environ["OPENAI_API_KEY"] = api_key
        vector_tool, summary_tool = get_doc_tools("swiss_bank_job.pdf", "swissbankjob")
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
        agent_worker = FunctionCallingAgentWorker.from_tools(
            [vector_tool, summary_tool],
            llm=llm,
            verbose=False
        )
        agent = AgentRunner(agent_worker)
        return agent

    agent = load_tools_and_agent(st.session_state['OPENAI_API_KEY'])

    st.markdown("---")
    st.subheader("Ask questions about the Swiss bank AI/NLP scientist job!")
    
    # 세션 상태 초기화
    if 'question_submitted' not in st.session_state:
        st.session_state.question_submitted = False
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    question = st.text_input("Your question:", key="question_input", on_change=lambda: setattr(st.session_state, 'question_submitted', True))
    
    # 엔터키 입력 감지 및 질문 처리
    if st.session_state.question_submitted and question and question != st.session_state.current_question:
        st.session_state.current_question = question
        st.session_state.question_submitted = False
        
        with st.spinner("Thinking..."):
            response = agent.chat(question)
            st.markdown(f"**🗣️ Question:** {question}")
            st.markdown(f"**💬 Answer:** {response.response}")
            sources = getattr(response, 'sources', None)
            if sources:
                pages = set()
                for i, src in enumerate(sources):
                    raw_output = getattr(src, "raw_output", None)
                    if raw_output is None:
                        continue
                    meta_dict = getattr(raw_output, "source_nodes", [])
                    for node_with_score in meta_dict:
                        node = node_with_score.node
                        if hasattr(node, "metadata"):
                            md = node.metadata
                            if "page_label" in md:
                                pages.add(md["page_label"])
                if pages:
                    st.info(f"📌 Source page(s): {', '.join(sorted(pages))}")
            else:
                st.warning("⚠️ No source found for this answer — please verify accuracy (possible hallucination).")
else:
    st.info("Please enter your OpenAI API key to start.") 