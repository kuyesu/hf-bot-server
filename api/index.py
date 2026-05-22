import os
import json
import asyncio
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient
import openai
import anthropic
import aiosmtplib

app = FastAPI(title="hf-bot Standard Gateway")

# ==========================================
# 🗄️ ENVIRONMENT SETUP & MONGO CONNECT
# ==========================================
MONGO_URI = os.getenv("MONGODB_URI")
XAI_API_KEY = os.getenv("XAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_CODEX_KEY = os.getenv("OPENAI_API_KEY")

# Gmail SMTP Credentials (Option A)
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_PASS")

if not MONGO_URI:
    raise RuntimeError("MONGODB_URI environment variable is missing!")

db_client = AsyncIOMotorClient(MONGO_URI)
db = db_client["hf_bot_analytics"]
logs_collection = db["chat_telemetry"]
shares_collection = db["shares_telemetry"]

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
    


# ==========================================
# ✉️ SHARED OUTBOUND SERVICE EMAIL ROUTE (aiosmtplib)
# ==========================================
@app.post("/v1/share")
async def send_developer_share_email(request: Request, x_client_uuid: str = Header(None)):
    user_ip = get_client_ip(request)
    client_id = x_client_uuid or user_ip
    
    body = await request.json()
    recipient_email = body.get("recipient_email")
    share_link = body.get("share_link", "https://github.com/your-username/hf-bot")
    sender_name = body.get("sender_name")  # 🌟 Dynamic structural data fetch
    
    if not recipient_email:
        raise HTTPException(status_code=400, detail="Missing required 'recipient_email' parameter.")
        
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        raise HTTPException(status_code=500, detail="Server SMTP gateway credentials are unconfigured.")

    # 🎨 Adapt messaging semantics based on personalization parameters
    if sender_name:
        subject_line = f"🚀 {sender_name} invited you to collaborate on hf-bot"
        greeting_copy = f"Your colleague, <strong>{sender_name}</strong>, has invited you to check out <code>hf-bot</code>"
    else:
        subject_line = "🚀 Code Collaboration Workspace Invitation: hf-bot"
        greeting_copy = "Another programmer has invited you to check out <code>hf-bot</code>"

    # Build the Custom HTML Developer Email Template using our dynamic copies
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; color: #f8fafc; padding: 24px; }}
            .card {{ background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 32px; max-width: 550px; margin: 0 auto; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3); }}
            .logo {{ font-size: 24px; font-weight: bold; color: #06b6d4; margin-bottom: 16px; text-align: center; }}
            p {{ color: #cbd5e1; line-height: 1.6; font-size: 15px; }}
            .btn-container {{ text-align: center; margin: 28px 0; }}
            .btn {{ background-color: #06b6d4; color: #0f172a !important; font-weight: bold; text-decoration: none; padding: 12px 24px; border-radius: 6px; font-size: 14px; display: inline-block; transition: background 0.2s; }}
            .footer {{ font-size: 12px; color: #64748b; text-align: center; margin-top: 24px; border-top: 1px solid #334155; padding-top: 16px; }}
            code {{ background-color: #0f172a; color: #a7f3d0; padding: 2px 6px; border-radius: 4px; font-family: monospace; }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="logo">🤗 hf-bot Workspace Container</div>
            <p>Hey Dev!</p>
            <p>{greeting_copy}—a local sandbox terminal utility built for validating structural Hugging Face repository models, infrastructure metrics, and automating code tasks.</p>
            <div class="btn-container">
                <a href="{share_link}" class="btn">Examine the Project Workspace</a>
            </div>
            <p>You can launch it right away using your terminal commands to gain full visibility into weight allocations, local file contexts, and automated execution diagnostics.</p>
            <div class="footer">
                Sent safely from hf-bot Server Framework • Tracking ID: {client_id}
            </div>
        </div>
    </body>
    </html>
    """

    message = MIMEMultipart("alternative")
    message["From"] = f"hf-bot Utility Workspace <{GMAIL_USER}>"
    message["To"] = recipient_email
    message["Subject"] = subject_line
    
    message.attach(MIMEText(f"Check out hf-bot, the local agent interface utility framework at: {share_link}", "plain"))
    message.attach(MIMEText(html_template, "html"))

    try:
        await aiosmtplib.send(
            message,
            hostname="smtp.gmail.com",
            port=587,
            username=GMAIL_USER,
            password=GMAIL_APP_PASSWORD,
            start_tls=True,
            timeout=15
        )
        
        # Log to MongoDB including the sender identity signature properties if tracked
        share_document = {
            "timestamp": datetime.utcnow(),
            "sender_client_uuid": client_id,
            "sender_ip_origin": user_ip,
            "sender_name_provided": sender_name,
            "recipient_target_email": recipient_email,
            "resource_link_shared": share_link,
            "status": "delivered"
        }
        await shares_collection.insert_one(share_document)
        return {"status": "success", "recipient_email": recipient_email, "shared_link": share_link}
        
    except Exception as e:
        await shares_collection.insert_one({
            "timestamp": datetime.utcnow(),
            "sender_client_uuid": client_id,
            "status": "failed",
            "error_log": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Inbound Mail Delivery Fault: {str(e)}")