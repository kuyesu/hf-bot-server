import os
import asyncio
from datetime import datetime
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient
import openai
import anthropic

app = FastAPI(title="hf-bot Advanced Multi-Provider Proxy")

# ==========================================
# 🗄️ SECRET KEYS & DATABASE CONNECTIONS
# ==========================================
MONGO_URI = os.getenv("MONGO_URI")
XAI_API_KEY = os.getenv("XAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_CODEX_KEY = os.getenv("OPENAI_API_KEY")  # For OpenAI's Codex ecosystem

if not MONGO_URI:
    raise RuntimeError("CRITICAL: MONGO_URI variable is missing!")

# Database handles
db_client = AsyncIOMotorClient(MONGO_URI)
db = db_client["hf_bot_analytics"]
logs_collection = db["chat_telemetry"]
limits_collection = db["daily_usage"]

# Initialize Multi-SDK client engines
xai_client = openai.OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_CODEX_KEY)
anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown-origin"

# ==========================================
# 🌊 ROUTE HANDLER
# ==========================================
@app.post("/v1/chat/completions")
async def proxy_stream_completion(request: Request, x_client_uuid: str = Header(None)):
    user_ip = get_client_ip(request)
    client_id = x_client_uuid or user_ip
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # 🛑 RATE LIMIT CHECKER
    usage_doc = await limits_collection.find_one_and_update(
        {"client_id": client_id, "date": today},
        {"$inc": {"count": 1}},
        upsert=True,
        return_document=True
    )
    if usage_doc and usage_doc.get("count", 0) > 50:
        raise HTTPException(status_code=429, detail="Shared limit hit (50/50 chats max daily).")

    # Parse Payload Configurations
    body = await request.json()
    messages = body.get("messages", [])
    last_user_message = messages[-1]["content"] if messages else ""
    requested_model = body.get("model", "grok-2").lower()

    # Determine Routing Matrix Base
    provider = "xai"
    if "claude" in requested_model or "anthropic" in requested_model:
        provider = "anthropic"
        # Map to a modern production flagship model if unassigned
        if "claude" not in requested_model:
            body["model"] = "claude-3-5-sonnet"
    elif "codex" in requested_model or "gpt-" in requested_model:
        provider = "openai_codex"
        if "gpt-" not in requested_model:
            body["model"] = "gpt-5.4" # Fallback to standard OpenAI flagship coding framework

    # 📝 WRITE USER PROMPT METRICS TO DB IMMEDIATELY
    log_doc = {
        "timestamp": datetime.utcnow(),
        "client_id": client_id,
        "user_ip": user_ip,
        "provider": provider,
        "model_used": body["model"],
        "input_message": last_user_message,
        "output_response_accumulated": "PENDING_STREAM_COMPLETION"
    }
    inserted_log = await logs_collection.insert_one(log_doc)
    log_id = inserted_log.inserted_id

    # ==========================================
    # 🏎️ PROVIDER ROUTING MATRIX
    # ==========================================
    try:
        if provider == "anthropic":
            # Translate standard chat payload syntax to match Anthropic parameters
            anthropic_messages = []
            system_prompt = ""
            for m in messages:
                if m["role"] == "system":
                    system_prompt = m["content"]
                else:
                    anthropic_messages.append({"role": m["role"], "content": m["content"]})

            async def stream_anthropic():
                accumulated_response = ""
                try:
                    async with anthropic_client.messages.stream(
                        model=body["model"],
                        max_tokens=4096,
                        system=system_prompt,
                        messages=anthropic_messages
                    ) as stream:
                        async for text in stream.text_stream:
                            accumulated_response += text
                            # Re-wrap back into OpenAI formatting style so your client works seamlessly
                            chunk_data = {
                                "choices": [{"delta": {"content": text}}]
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(
                        logs_collection.update_one({"_id": log_id}, {"$set": {"output_response_accumulated": accumulated_response}})
                    )

            return StreamingResponse(stream_anthropic(), media_type="text/event-stream")

        else:
            # Route out directly using Standard OpenAI SDK Protocol (Works for both xAI and OpenAI Codex)
            active_client = openai_client if provider == "openai_codex" else xai_client
            response = active_client.chat.completions.create(**body)

            def stream_openai_style():
                accumulated_response = ""
                try:
                    for chunk in response:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            accumulated_response += delta.content
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(
                        logs_collection.update_one({"_id": log_id}, {"$set": {"output_response_accumulated": accumulated_response}})
                    )

            return StreamingResponse(stream_openai_style(), media_type="text/event-stream")

    except Exception as e:
        await limits_collection.update_one({"client_id": client_id, "date": today}, {"$inc": {"count": -1}})
        await logs_collection.update_one({"_id": log_id}, {"$set": {"output_response_accumulated": f"ERROR: {str(e)}"} })
        raise HTTPException(status_code=500, detail=f"Handshake failed: {str(e)}")