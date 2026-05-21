import os
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient
import openai
import anthropic

app = FastAPI(title="hf-bot Standard Gateway")

# ==========================================
# 🗄️ ENVIRONMENT SETUP & MONGO CONNECT
# ==========================================
MONGO_URI = os.getenv("MONGODB_URI")
XAI_API_KEY = os.getenv("XAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_CODEX_KEY = os.getenv("OPENAI_API_KEY")

if not MONGO_URI:
    raise RuntimeError("MONGODB_URI environment variable is missing!")

db_client = AsyncIOMotorClient(MONGO_URI)
db = db_client["hf_bot_analytics"]
logs_collection = db["chat_telemetry"]

# Standardized API Clients
xai_client = openai.OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_CODEX_KEY)
anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown-origin"

# ==========================================
# 🌊 ROUTE HANDLER (Restoring clean streams)
# ==========================================
@app.post("/v1/chat/completions")
async def proxy_stream_completion(request: Request, x_client_uuid: str = Header(None)):
    user_ip = get_client_ip(request)
    client_id = x_client_uuid or user_ip

    body = await request.json()
    messages = body.get("messages", [])
    raw_model = str(body.get("model", "grok-4.3")).lower()
    last_user_message = messages[-1]["content"] if messages else ""

    # Normalize Provider and Models
    if "claude" in raw_model or "anthropic" in raw_model:
        provider = "anthropic"
        resolved_model = "claude-sonnet-4-6" if "opus" not in raw_model else "claude-opus-4-7"
    elif "codex" in raw_model or "gpt-" in raw_model:
        provider = "openai_codex"
        resolved_model = raw_model if "gpt-" in raw_model else "gpt-5.4"
    else:
        provider = "xai"
        resolved_model = "grok-4.3" if "grok" in raw_model or raw_model == "" else raw_model

    # Log telemetry instantly before data-stream opens
    log_doc = {
        "timestamp": datetime.utcnow(),
        "client_id": client_id,
        "user_ip": user_ip,
        "provider": provider,
        "model_used": resolved_model,
        "input_message": last_user_message,
        "output_response_accumulated": "PENDING_STREAM"
    }
    inserted_log = await logs_collection.insert_one(log_doc)
    log_id = inserted_log.inserted_id

    try:
        # --- PATH A: ANTHROPIC SDK TRANSACTION ---
        if provider == "anthropic":
            anthropic_messages = []
            system_prompt = ""
            for m in messages:
                if m.get("role") == "system":
                    system_prompt = m.get("content", "")
                else:
                    anthropic_messages.append({
                        "role": m.get("role", "user"),
                        "content": m.get("content", "")
                    })

            async def stream_anthropic():
                accumulated_text = ""
                try:
                    async with anthropic_client.messages.stream(
                        model=resolved_model,
                        max_tokens=4096,
                        system=system_prompt if system_prompt else None,
                        messages=anthropic_messages
                    ) as stream:
                        async for text in stream.text_stream:
                            accumulated_text += text
                            # 🎯 CRITICAL: Format chunks exactly like OpenAI structure for your CLI script
                            chunk = {
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                    
                    # End of stream signifier
                    done_chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    yield f"data: {json.dumps(done_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    # Sync log record tracking
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(
                        logs_collection.update_one(
                            {"_id": log_id},
                            {"$set": {"output_response_accumulated": accumulated_text}}
                        )
                    )

            return StreamingResponse(stream_anthropic(), media_type="text/event-stream")

        # --- PATH B: OPENAI / XAI STANDARD PROTOCOL ---
        else:
            openai_payload = {
                "model": resolved_model,
                "messages": messages,
                "stream": True,
                "tools": body.get("tools") if "tools" in body else None,
                "tool_choice": body.get("tool_choice") if "tool_choice" in body else None
            }
            # Clean up empty values to keep upstream schemas valid
            openai_payload = {k: v for k, v in openai_payload.items() if v is not None}
            
            if "temperature" in body:
                openai_payload["temperature"] = body["temperature"]

            active_client = openai_client if provider == "openai_codex" else xai_client
            response = active_client.chat.completions.create(**openai_payload)

            def stream_openai():
                accumulated_text = ""
                try:
                    for chunk in response:
                        delta = chunk.choices[0].delta if chunk.choices else None
                        if delta and hasattr(delta, "content") and delta.content:
                            accumulated_text += delta.content
                        
                        # Yield the raw compliant object directly to your local UI tool
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(
                        logs_collection.update_one(
                            {"_id": log_id},
                            {"$set": {"output_response_accumulated": accumulated_text}}
                        )
                    )

            return StreamingResponse(stream_openai(), media_type="text/event-stream")

    except Exception as e:
        await logs_collection.update_one(
            {"_id": log_id},
            {"$set": {"output_response_accumulated": f"ROUTING_ERROR: {str(e)}"}}
        )
        raise HTTPException(status_code=500, detail=str(e))