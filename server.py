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
# Vercel Environment Variables (Configure these in your Vercel Dashboard)
XAI_API_KEY = os.getenv("PROD_XAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")  # e.g., mongodb+srv://<user>:<pwd>@cluster.mongodb.net/myDb

if not MONGO_URI:
    raise RuntimeError("CRITICAL: MONGO_URI environment variable is missing!")

# Initialize Async MongoDB Engine
db_client = AsyncIOMotorClient(MONGO_URI)
db = db_client["hf_bot_analytics"]
logs_collection = db["chat_telemetry"]
limits_collection = db["daily_usage"]

openai_client = openai.OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)

# ==========================================
# 🛰️ HELPER METHOD UTILITIES
# ==========================================
def get_client_ip(request: Request) -> str:
    """Extracts true remote user IP to locate region profiles under proxies."""
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

    # 🛑 GUARDRAIL: Atomically find or increment the user's daily usage in MongoDB
    usage_doc = await limits_collection.find_one_and_update(
        {"client_id": client_id, "date": today},
        {"$inc": {"count": 1}},
        upsert=True,
        return_document=True
    )

    # Verify usage parameters against the daily allowance threshold
    if usage_doc and usage_doc.get("count", 0) > 50:
        raise HTTPException(
            status_code=429,
            detail="Shared sandbox quota exceeded (50/50 chats used today). Please bind a personal token."
        )

    # Parse inbound request structures
    body = await request.json()
    messages = body.get("messages", [])
    last_user_message = messages[-1]["content"] if messages else "Empty Prompt Payload"

    # Create placeholder document token trace in MongoDB to capture user inputs
    log_doc = {
        "timestamp": datetime.utcnow(),
        "client_id": client_id,
        "user_ip": user_ip,
        "input_message": last_user_message,
        "output_response_accumulated": ""
    }
    # Insert asynchronously to avoid blocking the network thread connection
    inserted_log = await logs_collection.insert_one(log_doc)
    log_id = inserted_log.inserted_id

    # Securely build streaming request connection pipeline out to xAI servers
    try:
        response = openai_client.chat.completions.create(**body)

        async def stream_chunks():
            accumulated_response = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    accumulated_response += delta.content
                yield f"data: {chunk.model_dump_json()}\n\n"
            
            yield "data: [DONE]\n\n"

            # 📝 UPDATE TELEMETRY: Save the final response text back into MongoDB
            await logs_collection.update_one(
                {"_id": log_id},
                {"$set": {"output_response_accumulated": accumulated_response}}
            )

        return StreamingResponse(stream_chunks(), media_type="text/event-stream")

    except Exception as e:
        # If upstream server collapses, decrement usage count so user isn't penalized
        await limits_collection.update_one(
            {"client_id": client_id, "date": today},
            {"$inc": {"count": -1}}
        )
        raise HTTPException(status_code=500, detail=f"Proxy handshake failed: {str(e)}")
