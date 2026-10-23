import streamlit as st
import requests

st.set_page_config(
    page_title="F1 AI Analytics Assistant",
    page_icon="🏁",
    layout="centered"
)

# Custom styling for F1 theme
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #ffffff; }
    h1 { color: #e10600; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🏁 F1 LLM-RAG + Graph + Multi-Model Engine</h1>", unsafe_allow_html=True)
st.caption("🚀 Powered by FastAPI, Local Ollama Llama 3, Hybrid Vector/Graph RAG, and 6 Parallel ML Engines.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt input
if prompt := st.chat_input("Ask an F1 question (e.g., Will Max Verstappen win the next podium?)..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Executing 6 parallel ML models & searching knowledge graph..."):
            try:
                response = requests.post(
                    "http://localhost:8000/chat",
                    json={"prompt": prompt, "run_training": True},
                    timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
                    winning_model = data.get("winning_model", "unknown").upper()
                    score = data.get("model_score", 0.0) * 100
                    intent = data.get("detected_intent", {})
                    llm_output = data.get("llm_response", "No response generated.")

                    # Format assistant output
                    reply_text = f"""### 🏆 Deployed Model: **{winning_model}** (Accuracy: {score:.1f}%)
**Target Identified:** `{intent.get('target')}` (*Task: {intent.get('task')}*)

---
{llm_output}
"""
                else:
                    reply_text = f"⚠️ **API Error**: Server responded with status code {response.status_code}"
            except Exception as e:
                reply_text = f"❌ **Connection Error**: Could not reach FastAPI backend at `http://localhost:8000`. Make sure it's running (`uvicorn api.app:app --reload`). Details: `{e}`"

            st.markdown(reply_text)
            st.session_state.messages.append({"role": "assistant", "content": reply_text})