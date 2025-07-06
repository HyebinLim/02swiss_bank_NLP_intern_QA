import streamlit as st
from helper import get_openai_api_key
from utils import get_doc_tools
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
import os
import re

# 페이지 설정
st.set_page_config(
    page_title="Swiss Bank AI/NLP Job Q&A",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 앱 설명 표시 ---
# CSS로 네모상자 스타일 적용
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
* {
    font-family: 'Poppins', sans-serif;
}
.custom-box {
    background-color: #e6f3ff;
    padding: 5px 20px;
    border-radius: 10px;
    margin: 10px 0;
    font-family: 'Poppins', sans-serif;
    display: flex;
    align-items: center;
    min-height: 50px;
}
.custom-box h3 {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    margin: 0;
    color: #2c3e50;
}
.custom-desc-list {
    font-family: 'Poppins', sans-serif;
}
.custom-desc-list li {
    margin-bottom: 2px;
    margin-top: 2px;
    font-family: 'Poppins', sans-serif;
    font-weight: 400;
    color: #34495e;
    font-size: 14px;
}
h2 {
    font-family: 'Poppins', sans-serif;
    font-size: 16px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-box"><h3>Want a Swiss Bank AI/NLP job?</h3></div>', unsafe_allow_html=True)
st.markdown('''
<div class="custom-desc-list">
<ul>
<li>Ask me anything about my journey to become an AI/NLP scientist at a bank in Switzerland during my master's at University of Zurich.</li>
<li>Answers based on <a href="https://blog.naver.com/imyourbest" target="_blank">89% 직장인 일지</a></li>
</ul>
</div>
''', unsafe_allow_html=True)

# --- API 키 입력 ---
if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state['OPENAI_API_KEY'] = ''

# API 키 재설정 버튼은 Q&A 시스템 로드 후에 표시됩니다

api_key = st.text_input(
    "🔑 Enter your OpenAI API key:",
    type="password",
    value=st.session_state['OPENAI_API_KEY'],
    key="api_key_input"
)

# API 키가 변경되었는지 확인
if api_key and api_key != st.session_state['OPENAI_API_KEY']:
    st.session_state['OPENAI_API_KEY'] = api_key
    st.session_state['agent_loaded'] = False
    st.success("✅ API key updated! Please wait for the system to load...")

def translate_to_english(question, api_key):
    """한국어 질문을 영어로 번역"""
    try:
        import os
        # proxies 설정 제거 (OpenAI 1.56.0+ 호환성)
        proxy_vars = ["OPENAI_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
        for var in proxy_vars:
            if var in os.environ:
                del os.environ[var]
            
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful translator specializing in job-related questions. Translate the user's question to English, focusing on job search, career, salary, requirements, and workplace topics. If the question is already in English, return it as is. Only return the translated question, nothing else. Be precise with job-related terminology."},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Translation failed: {str(e)}. Using original question.")
        return question

def extract_source_pages(response):
    """응답에서 소스 페이지를 추출하는 개선된 함수 - 상위 3-5개만 표시"""
    try:
        sources = getattr(response, 'sources', None)
        if not sources:
            return None
            
        pages = set()
        for src in sources:
            raw_output = getattr(src, "raw_output", None)
            if raw_output is None:
                continue
                
            # source_nodes가 있는 경우
            meta_dict = getattr(raw_output, "source_nodes", [])
            if meta_dict:
                for node_with_score in meta_dict:
                    node = node_with_score.node
                    if hasattr(node, "metadata"):
                        md = node.metadata
                        if "page_label" in md:
                            pages.add(md["page_label"])
            
            # 직접 metadata가 있는 경우
            if hasattr(raw_output, "metadata"):
                md = raw_output.metadata
                if "page_label" in md:
                    pages.add(md["page_label"])
        
        if pages:
            # 페이지 번호를 숫자로 변환하여 정렬
            sorted_pages = sorted([int(p) if p.isdigit() else 0 for p in pages])
            # 상위 5개만 반환
            top_pages = sorted_pages[:5]
            return [str(p) for p in top_pages]
        return None
    except Exception as e:
        st.warning(f"Error extracting source pages: {str(e)}")
        return None

# --- Q&A 기능 준비 ---
if st.session_state['OPENAI_API_KEY']:
    # 문서 도구 및 에이전트 준비 (캐싱)
    @st.cache_resource(show_spinner=True)
    def load_tools_and_agent(api_key):
        try:
            import os
            os.environ["OPENAI_API_KEY"] = api_key
            # proxies 설정 제거 (OpenAI 1.56.0+ 호환성)
            proxy_vars = ["OPENAI_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
            for var in proxy_vars:
                if var in os.environ:
                    del os.environ[var]
            
            # PDF 파일 존재 확인 (원본 사용 - 텍스트 보존)
            pdf_path = "swiss_bank_job.pdf"
            if not os.path.exists(pdf_path):
                st.error(f"PDF file not found: {pdf_path}")
                return None
                
            vector_tool, summary_tool = get_doc_tools(pdf_path, "swissbankjob")
            llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
            agent_worker = FunctionCallingAgentWorker.from_tools(
                [vector_tool, summary_tool],
                llm=llm,
                verbose=False
            )
            agent = AgentRunner(agent_worker)
            return agent
        except Exception as e:
            st.error(f"Error loading tools and agent: {str(e)}")
            return None

    # 에이전트 로딩 상태 확인
    if 'agent_loaded' not in st.session_state:
        st.session_state['agent_loaded'] = False

    # API 키 재설정 시 캐시 클리어
    if st.button("🔄 Reset API Key"):
        st.session_state['OPENAI_API_KEY'] = ''
        st.session_state['agent_loaded'] = False
        load_tools_and_agent.clear()
        st.rerun()

    if not st.session_state['agent_loaded']:
        with st.spinner("Loading Q&A system..."):
            agent = load_tools_and_agent(st.session_state['OPENAI_API_KEY'])
            if agent is not None:
                st.session_state['agent'] = agent
                st.session_state['agent_loaded'] = True
                st.success("✅ Q&A system loaded successfully!")
            else:
                st.error("❌ Failed to load Q&A system. Please check your API key and try again.")
    else:
        agent = st.session_state['agent']

    if st.session_state['agent_loaded'] and agent is not None:
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
            
            try:
                with st.spinner("Processing your question..."):
                    # 한국어 질문을 영어로 번역
                    translated_question = translate_to_english(question, st.session_state['OPENAI_API_KEY'])
                    
                    # 번역된 질문으로 응답 생성
                    response = agent.chat(translated_question)
                    
                    # 결과 표시
                    st.markdown(f"**🗣️ Question:** {question}")
                    st.markdown(f"**💬 Answer:** {response.response}")
                    
                    # 응답 내용을 분석해서 실제로 정보가 있는지 확인
                    answer_text = response.response.lower()
                    no_info_keywords = ['no information', 'not found', 'not available', 'not mentioned', 'not provided', 'not specified', 'not stated', 'not given', 'not included', 'not listed', 'not detailed', 'not described', 'not outlined', 'not covered', 'not addressed', 'not discussed', 'not revealed', 'not disclosed', 'not shared', 'not indicated']
                    
                    has_information = not any(keyword in answer_text for keyword in no_info_keywords)
                    
                    # 소스 페이지 추출 및 표시 (정보가 있을 때만)
                    if has_information:
                        source_pages = extract_source_pages(response)
                        if source_pages:
                            st.info(f"📌 Source page(s): {', '.join(source_pages)}")
                            st.info("📄 Data source: [GitHub PDF](https://github.com/HyebinLim/02swiss_bank_NLP_intern_QA/blob/master/swiss_bank_job.pdf)")
                        else:
                            st.info("📌 Answer generated from document content (specific pages not available)")
                            st.info("📄 Data source: [GitHub PDF](https://github.com/HyebinLim/02swiss_bank_NLP_intern_QA/blob/master/swiss_bank_job.pdf)")
                    else:
                        st.warning("⚠️ No information found in the document for this question.")
                            
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
    elif not st.session_state['agent_loaded']:
        st.info("Please wait while the Q&A system is loading...")
# API 키 입력 안내 메시지 제거 