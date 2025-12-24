"""
Cookbook: Streaming Chatbot with FastAPI + SSE
Run: uvicorn examples.cookbook.streaming_chatbot:app --reload

Dependencies:
    pip install fastapi uvicorn
"""
import os
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from aiclient import Client

load_dotenv()
app = FastAPI()
client = Client()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
    <body>
        <h1>Streaming Chatbot ⚡</h1>
        <input type="text" id="prompt" placeholder="Ask me something..." style="width: 300px;">
        <button onclick="send()">Send</button>
        <div id="response" style="margin-top: 20px; white-space: pre-wrap; font-family: monospace;"></div>

        <script>
            async function send() {
                const prompt = document.getElementById('prompt').value;
                const div = document.getElementById('response');
                div.innerHTML = "";
                
                const response = await fetch(`/chat?prompt=${encodeURIComponent(prompt)}`);
                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const {value, done} = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    div.innerHTML += chunk;
                }
            }
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(HTML_TEMPLATE)

@app.get("/chat")
async def chat(prompt: str):
    async def event_generator():
        # Using aiclient's async stream
        # Using a relatively fast model for demo
        model = "gpt-4o-mini" 
        
        async for chunk in client.chat(model).stream_async(prompt):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/plain")
