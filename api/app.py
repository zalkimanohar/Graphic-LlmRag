import os
import json
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Internal module imports
from ml_engine.autolabel import AutoLabeler
from ml_engine.trainer import ParallelMLTrainer
from ml_engine.evaluator import MLEvaluator
from graph_store.neo4j_connector import F1GraphStore
from rag_engine.vector_store import F1VectorStore
from rag_engine.retriever import F1HybridRetriever

try:
    import ollama
except ImportError:
    ollama = None

app = FastAPI(title="F1 Analytics LLM-RAG + Graph + ML Engine", version="1.0.0")

# Enable CORS Middleware so Swagger UI / Browser JS can freely fetch API endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
autolabeler = AutoLabeler()
vector_store = F1VectorStore()
graph_store = F1GraphStore(data_dir="../data")
hybrid_retriever = F1HybridRetriever(vector_store, graph_store)

class QueryRequest(BaseModel):
    prompt: str
    run_training: bool = False

@app.get("/", response_class=HTMLResponse)
def get_chat_ui():
    """
    Serves an interactive chat dashboard directly in the browser.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>F1 AI Analytics Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #ffffff; margin: 0; padding: 20px; display: flex; flex-direction: column; height: 90vh; }
            h1 { color: #e10600; text-align: center; font-size: 24px; margin-bottom: 20px; }
            #chat-box { flex: 1; border: 1px solid #333; padding: 20px; overflow-y: auto; background-color: #1e1e1e; border-radius: 8px; margin-bottom: 20px; }
            .message { margin-bottom: 15px; padding: 12px 18px; border-radius: 6px; line-height: 1.5; white-space: pre-wrap; }
            .user { background-color: #2b394a; text-align: right; margin-left: 25%; color: #64b5f6; }
            .bot { background-color: #2d2d2d; border-left: 4px solid #e10600; margin-right: 25%; }
            .meta { font-size: 11px; color: #aaa; margin-bottom: 6px; font-weight: bold; }
            .input-area { display: flex; gap: 10px; }
            input[type="text"] { flex: 1; padding: 12px; border-radius: 6px; border: 1px solid #444; background-color: #2a2a2a; color: white; font-size: 14px; }
            button { padding: 12px 24px; background-color: #e10600; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; }
            button:hover { background-color: #b30500; }
        </style>
    </head>
    <body>
        <h1>🏁 F1 LLM-RAG + Graph + Multi-Model Engine</h1>
        <div id="chat-box"></div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="Ask an F1 question (e.g., Will Max Verstappen win the next podium?)..." onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">Send Query</button>
        </div>

        <script>
            async function sendMessage() {
                const inputEl = document.getElementById('user-input');
                const chatBox = document.getElementById('chat-box');
                const prompt = inputEl.value.trim();
                
                if (!prompt) return;

                // Render User Query
                chatBox.innerHTML += `<div class="message user"><div class="meta">You</div>${prompt}</div>`;
                inputEl.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;

                // Render Loading Placeholder
                const loadingId = 'loading-' + Date.now();
                chatBox.innerHTML += `<div class="message bot" id="${loadingId}"><div class="meta">F1 AI System</div><i>Executing 6 parallel ML models & searching knowledge graph...</i></div>`;
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt: prompt, run_training: true })
                    });
                    const data = await response.json();
                    
                    const botMsgEl = document.getElementById(loadingId);
                    botMsgEl.innerHTML = `
                        <div class="meta">F1 AI System | 🏆 Deployed Model: <b>${data.winning_model.toUpperCase()}</b> (Accuracy: ${(data.model_score * 100).toFixed(1)}%)</div>
                        <div><b>Target Identified:</b> ${data.detected_intent.target} (${data.detected_intent.task})</div>
                        <hr style="border: 0.5px solid #444; margin: 10px 0;">
                        <div>${data.llm_response || 'No LLM output received.'}</div>
                    `;
                } catch (err) {
                    document.getElementById(loadingId).innerHTML = `<div class="meta">Error</div>Failed to communicate with API server.`;
                }
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    try:
        user_prompt = request.prompt
        print(f"\n[API] Received prompt: {user_prompt}")

        # 1. Auto-detect target and task from prompt
        intent = autolabeler.detect_target_from_prompt(user_prompt)
        target_col = intent["target"]
        task_type = intent["task"]

        # 2. Trigger ML training / prediction dynamically based on task type
        model_score = 0.88
        winner_model_name = "xgboost"
        registry_meta = "ml_engine/registry/model_metadata.json"
        
        # Always run training when requested or ensure task-based evaluation
        if request.run_training or not os.path.exists(registry_meta):
            trainer = ParallelMLTrainer(data_dir="../data")
            results, _, _ = trainer.train_all_models(target_col=target_col, task=task_type)
            evaluator = MLEvaluator(registry_dir="ml_engine/registry")
            winner_model_name = evaluator.evaluate_and_deploy(results)
            if os.path.exists(registry_meta):
                with open(registry_meta, "r") as f:
                    meta = json.load(f)
                    model_score = meta.get("performance_score", 0.88)
        else:
            with open(registry_meta, "r") as f:
                meta = json.load(f)
                winner_model_name = meta.get("winner_model", "xgboost")
                model_score = meta.get("performance_score", 0.88)

        # 3. Retrieve Hybrid Context (Vector + Graph)
        context = hybrid_retriever.retrieve_context(user_prompt)

        # 4. Construct Clean RAG prompt payload
        rag_prompt = f"""You are an elite F1 DataOps and Machine Learning Assistant. 
Analyze the user request using the retrieved context and the winning machine learning model prediction.

[ML Engine Insights]
- Target Variable: {target_col}
- Task Type: {task_type}
- Winning Model: {winner_model_name.upper()}
- Model Validation Score: {model_score:.4f}

[Retrieved Hybrid Context]
{context}

[User Prompt]
{user_prompt}

Provide a precise, expert F1 analysis combining the model results and context:"""

        # 5. Local LLM Generation via Ollama
        llm_response = ""
        if ollama:
            try:
                response = ollama.generate(model='llama3', prompt=rag_prompt)
                llm_response = response.get('response', '')
            except Exception as e:
                llm_response = f"[Ollama Notice]: Local runner offline or model not loaded ({e})."
        else:
            llm_response = "[Ollama Notice]: ollama python package not installed."

        return {
            "query": user_prompt,
            "detected_intent": intent,
            "winning_model": winner_model_name,
            "model_score": model_score,
            "context_summary": context[:300] + "...",
            "llm_response": llm_response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))