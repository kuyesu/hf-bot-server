import os
from datetime import datetime
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient
import openai

app = FastAPI(title="hf-bot Vercel Production Proxy")

# ==========================================
# 🗄️ DATABASE & CLOUD CONNECTIONS
# ==========================================
XAI_API_KEY = os.getenv("PROD_XAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("CRITICAL: MONGO_URI environment variable is missing!")

# Initialize Asynchronous Mongo Client
db_client = AsyncIOMotorClient(MONGO_URI)
db = db_client["hf_bot_analytics"]
logs_collection = db["chat_telemetry"]
limits_collection = db["daily_usage"]

openai_client = openai.OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)

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

    # 🛑 GUARDRAIL: Atomically check daily rate limits
    usage_doc = await limits_collection.find_one_and_update(
        {"client_id": client_id, "date": today},
        {"$inc": {"count": 1}},
        upsert=True,
        return_document=True
    )

    if usage_doc and usage_doc.get("count", 0) > 50:
        raise HTTPException(
            status_code=429,
            detail="Shared sandbox quota exceeded (50/50 chats used today). Please bind a personal token."
        )

    # Parse inbound request data payload
    body = await request.json()
    messages = body.get("messages", [])
    last_user_message = messages[-1]["content"] if messages else "Empty Prompt Payload"

    # Ensure we use a valid model name supported by your xAI console project tier
    # Force a reliable model identifier if the incoming body has it missing or custom-mapped
    incoming_model = body.get("model", "")
    if not incoming_model or "grok" not in incoming_model.lower():
        body["model"] = "grok-2" # 🌟 Fallback to standard base production Grok if unspecified
    
    # 📝 Write user prompt telemetry to database BEFORE starting streaming connection handles
    log_doc = {
        "timestamp": datetime.utcnow(),
        "client_id": client_id,
        "user_ip": user_ip,
        "input_message": last_user_message,
        "output_response_accumulated": "PENDING_STREAM_COMPLETION",
        "model_used": body["model"]
    }
    inserted_log = await logs_collection.insert_one(log_doc)
    log_id = inserted_log.inserted_id

    # Securely connect to upstream xAI servers
    try:
        # Pass the updated request body directly
        response = openai_client.chat.completions.create(**body)

        def stream_chunks():
            accumulated_response = ""
            try:
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        accumulated_response += delta.content
                    yield f"data: {chunk.model_dump_json()}\n\n"
                
                yield "data: [DONE]\n\n"
            finally:
                # 🔄 Write final streaming text block back into MongoDB Atlas
                import asyncio
                try:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(
                        logs_collection.update_one(
                            {"_id": log_id},
                            {"$set": {"output_response_accumulated": accumulated_response}}
                        )
                    )
                except Exception:
                    pass

        return StreamingResponse(stream_chunks(), media_type="text/event-stream")

    except Exception as e:
        # Revert counter if upstream handshake breaks
        await limits_collection.update_one(
            {"client_id": client_id, "date": today},
            {"$inc": {"count": -1}}
        )
        await logs_collection.update_one(
            {"_id": log_id},
            {"$set": {"output_response_accumulated": f"ERROR: {str(e)}"}}
        )
        raise HTTPException(status_code=500, detail=f"Proxy handshake failed: {str(e)}")