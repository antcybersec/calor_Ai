"""
FastAPI Server & Web Chat UI for CalorAI.
Provides REST API endpoints for meal logging, vision uploads, daily totals, memory retrieval,
and serves an interactive web UI.
"""

import os
import shutil
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from agent import run_agent_turn
from database import get_daily_totals, get_memories, clear_user_data, init_db

load_dotenv()
init_db()

app = FastAPI(
    title="CalorAI Conversational Agent API",
    description="WhatsApp-native meal logging agent with persistent memory & vision routing.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "calor_ai_uploads")
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@app.post("/api/chat")
def chat_endpoint(
    user_id: str = Form("web_user"),
    session_id: str = Form("web_session"),
    message: str = Form(""),
    image: Optional[UploadFile] = File(None)
):
    """
    Main Chat API endpoint. Accepts text message and optional uploaded meal image.
    """
    saved_image_path = None
    if image and image.filename:
        file_ext = os.path.splitext(image.filename)[1] or ".jpg"
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"upload_{user_id}_{os.urandom(4).hex()}{file_ext}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        saved_image_path = temp_file_path

    result = run_agent_turn(
        user_id=user_id,
        message_text=message,
        image_path=saved_image_path,
        session_id=session_id
    )

    return JSONResponse(result)

@app.get("/api/totals")
def get_totals_endpoint(user_id: str = Query("web_user"), date: Optional[str] = Query(None)):
    """Retrieve daily nutrition totals for user."""
    totals = get_daily_totals(user_id=user_id, query_date=date)
    return JSONResponse(totals)

@app.get("/api/memories")
def get_memories_endpoint(user_id: str = Query("web_user")):
    """Retrieve stored persistent user memory profile."""
    memories = get_memories(user_id=user_id)
    return JSONResponse({"user_id": user_id, "memories": memories})

@app.delete("/api/clear")
def clear_user_endpoint(user_id: str = Query("web_user")):
    """Clear database records for specified user."""
    clear_user_data(user_id)
    return JSONResponse({"status": "success", "message": f"Cleared data for user '{user_id}'"})

@app.get("/", response_class=HTMLResponse)
def serve_web_ui():
    """Serves the interactive web UI for CalorAI."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CalorAI — Conversational Meal Logging Agent</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        body { background-color: #0b1319; color: #e9edef; display: flex; height: 100vh; overflow: hidden; }
        
        .sidebar { width: 340px; background: #111b21; border-right: 1px solid #222d34; padding: 20px; display: flex; flex-direction: column; gap: 20px; }
        .sidebar-header { display: flex; align-items: center; gap: 12px; font-size: 20px; font-weight: 700; color: #25d366; }
        .card { background: #202c33; border-radius: 12px; padding: 16px; border: 1px solid #2a3942; }
        .card h3 { font-size: 14px; text-transform: uppercase; color: #8696a0; margin-bottom: 12px; letter-spacing: 0.5px; }
        .metric-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; }
        .metric-value { font-weight: 700; color: #00a884; }

        .main-chat { flex: 1; display: flex; flex-direction: column; background: #0b141a; }
        .chat-header { background: #202c33; padding: 16px 24px; border-bottom: 1px solid #222d34; display: flex; justify-content: space-between; align-items: center; }
        .chat-header h2 { font-size: 16px; font-weight: 600; color: #e9edef; }
        
        .messages-container { flex: 1; padding: 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; background-image: radial-gradient(#1f2c34 1px, transparent 1px); background-size: 20px 20px; }
        .message { max-width: 65%; padding: 12px 16px; border-radius: 12px; font-size: 14.5px; line-height: 1.4; word-wrap: break-word; }
        .message.user { align-self: flex-end; background: #005c4b; color: #e9edef; border-bottom-right-radius: 2px; }
        .message.assistant { align-self: flex-start; background: #202c33; color: #e9edef; border-bottom-left-radius: 2px; border: 1px solid #2a3942; }
        .meta-tag { font-size: 11px; opacity: 0.6; margin-top: 4px; text-align: right; }

        .input-area { background: #202c33; padding: 16px 24px; border-top: 1px solid #222d34; display: flex; gap: 12px; align-items: center; }
        .input-area input[type="text"] { flex: 1; background: #2a3942; border: none; padding: 14px 18px; border-radius: 8px; color: #fff; font-size: 15px; outline: none; }
        .file-label { background: #2a3942; color: #8696a0; padding: 12px; border-radius: 8px; cursor: pointer; display: flex; align-items: center; }
        .file-label:hover { background: #374853; color: #25d366; }
        .send-btn { background: #00a884; color: #111b21; border: none; padding: 14px 24px; border-radius: 8px; font-weight: 700; cursor: pointer; transition: 0.2s; }
        .send-btn:hover { background: #06cf9c; }
        #image-preview { font-size: 12px; color: #25d366; }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-header">
            <span>🥗</span> CalorAI Agent
        </div>
        <div class="card">
            <h3>Running Daily Totals</h3>
            <div class="metric-row"><span>Calories:</span><span class="metric-value" id="tot-cal">0 kcal</span></div>
            <div class="metric-row"><span>Protein:</span><span class="metric-value" id="tot-prot">0 g</span></div>
            <div class="metric-row"><span>Carbs:</span><span class="metric-value" id="tot-carbs">0 g</span></div>
            <div class="metric-row"><span>Fat:</span><span class="metric-value" id="tot-fat">0 g</span></div>
            <div class="metric-row"><span>Meals Logged:</span><span class="metric-value" id="tot-count">0</span></div>
        </div>
        <div class="card" style="flex: 1; overflow-y: auto;">
            <h3>Persistent User Memories</h3>
            <div id="memories-list" style="font-size: 13px; color: #8696a0; display: flex; flex-direction: column; gap: 8px;">
                Loading memories...
            </div>
        </div>
    </div>
    
    <div class="main-chat">
        <div class="chat-header">
            <h2>WhatsApp Conversational Logging Sandbox</h2>
            <button onclick="clearData()" style="background: transparent; border: 1px solid #ff4b4b; color: #ff4b4b; padding: 6px 12px; border-radius: 6px; cursor: pointer;">Clear Sandbox</button>
        </div>
        <div class="messages-container" id="chat">
            <div class="message assistant">
                Hey there! 👋 I'm your CalorAI assistant. Tell me what you ate (e.g. "had 2 rotis and dal") or upload a picture of your plate!
                <div class="meta-tag">System Ready</div>
            </div>
        </div>
        <div class="input-area">
            <label class="file-label" title="Attach Plate Photo">
                📷 <input type="file" id="img-input" accept="image/*" style="display:none;" onchange="previewImage()">
            </label>
            <span id="image-preview"></span>
            <input type="text" id="msg-input" placeholder="Type what you ate or ask 'how am I doing today?'..." onkeydown="if(event.key==='Enter') sendMessage()">
            <button class="send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const USER_ID = "web_demo_user";

        async function updateDashboard() {
            try {
                const tRes = await fetch(`/api/totals?user_id=${USER_ID}`);
                const totals = await tRes.json();
                document.getElementById('tot-cal').innerText = `${totals.total_calories} kcal`;
                document.getElementById('tot-prot').innerText = `${totals.total_protein_g} g`;
                document.getElementById('tot-carbs').innerText = `${totals.total_carbs_g} g`;
                document.getElementById('tot-fat').innerText = `${totals.total_fat_g} g`;
                document.getElementById('tot-count').innerText = totals.meal_count;

                const mRes = await fetch(`/api/memories?user_id=${USER_ID}`);
                const mData = await mRes.json();
                const mList = document.getElementById('memories-list');
                if(mData.memories && mData.memories.length > 0) {
                    mList.innerHTML = mData.memories.map(m => `
                        <div style="background: #111b21; padding: 8px; border-radius: 6px; border-left: 3px solid #00a884;">
                            <strong style="color:#e9edef;">${m.memory_key.replace('_', ' ').toUpperCase()}</strong><br>${m.memory_value}
                        </div>
                    `).join('');
                } else {
                    mList.innerHTML = '<i>No persistent memories stored yet.</i>';
                }
            } catch(e) { console.error(e); }
        }

        function previewImage() {
            const file = document.getElementById('img-input').files[0];
            document.getElementById('image-preview').innerText = file ? `📷 ${file.name}` : '';
        }

        async function sendMessage() {
            const input = document.getElementById('msg-input');
            const fileInput = document.getElementById('img-input');
            const text = input.value.trim();
            const file = fileInput.files[0];

            if(!text && !file) return;

            const chat = document.getElementById('chat');
            const userMsgDiv = document.createElement('div');
            userMsgDiv.className = 'message user';
            userMsgDiv.innerText = text + (file ? ` [Attached: ${file.name}]` : '');
            chat.appendChild(userMsgDiv);

            input.value = '';
            fileInput.value = '';
            document.getElementById('image-preview').innerText = '';
            chat.scrollTop = chat.scrollHeight;

            const formData = new FormData();
            formData.append('user_id', USER_ID);
            formData.append('session_id', 'web_session');
            formData.append('message', text);
            if(file) formData.append('image', file);

            const assistantMsgDiv = document.createElement('div');
            assistantMsgDiv.className = 'message assistant';
            assistantMsgDiv.innerText = "CalorAI is typing...";
            chat.appendChild(assistantMsgDiv);
            chat.scrollTop = chat.scrollHeight;

            try {
                const res = await fetch('/api/chat', { method: 'POST', body: formData });
                const data = await res.json();
                assistantMsgDiv.innerHTML = `${data.response} <div class="meta-tag">${data.latency_seconds}s latency</div>`;
                updateDashboard();
            } catch(err) {
                assistantMsgDiv.innerText = "Error processing request.";
            }
            chat.scrollTop = chat.scrollHeight;
        }

        async function clearData() {
            await fetch(`/api/clear?user_id=${USER_ID}`, { method: 'DELETE' });
            document.getElementById('chat').innerHTML = '<div class="message assistant">Sandbox cleared! Start typing.</div>';
            updateDashboard();
        }

        updateDashboard();
    </script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
