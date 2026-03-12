"""
server.py — HTTP Server + ngrok Tunnel for Vobiz Webhooks
==========================================================
Serves XML webhooks for Vobiz call handling and manages ngrok tunnel.
Starts both the HTTP server and the agent WebSocket server.
"""

import os
import sys
import asyncio
import logging
import threading
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import Response
from dotenv import load_dotenv
from pyngrok import ngrok, conf

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HTTP_PORT = int(os.getenv("HTTP_PORT", "5000"))
WS_PORT = int(os.getenv("AGENT_WS_PORT", "5001"))
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("server")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="Vobiz Voice Agent Server")

# Will be set after ngrok tunnel is established
NGROK_URL = None


@app.post("/answer")
async def answer_call(request: Request):
    """
    Vobiz calls this webhook when a call connects (inbound or outbound).
    Returns XML with a bidirectional Stream element pointing to the agent WebSocket.
    """
    form_data = await request.form()
    call_uuid = form_data.get("CallUUID", "unknown")
    from_number = form_data.get("From", "unknown")
    to_number = form_data.get("To", "unknown")
    direction = form_data.get("Direction", "unknown")

    logger.info(f"📞 Call connected — UUID={call_uuid}, From={from_number}, To={to_number}, Direction={direction}")

    # Build the WebSocket URL for the agent
    # ngrok HTTPS URL → convert to WSS
    ws_url = NGROK_URL.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/ws"

    # Return Vobiz XML with bidirectional Stream
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000" statusCallbackUrl="{NGROK_URL}/stream-status" statusCallbackMethod="POST">
        {ws_url}
    </Stream>
</Response>"""

    logger.info(f"Returning Stream XML → WebSocket URL: {ws_url}")
    return Response(content=xml_response, media_type="application/xml")


@app.post("/hangup")
async def hangup_call(request: Request):
    """Vobiz calls this when the call ends."""
    form_data = await request.form()
    call_uuid = form_data.get("CallUUID", "unknown")
    duration = form_data.get("Duration", "0")
    hangup_cause = form_data.get("HangupCause", "unknown")

    logger.info(f"📴 Call ended — UUID={call_uuid}, Duration={duration}s, Cause={hangup_cause}")
    return Response(content="OK", status_code=200)


@app.post("/stream-status")
async def stream_status(request: Request):
    """Vobiz sends stream lifecycle events here."""
    form_data = await request.form()
    event = form_data.get("Event", "unknown")
    stream_id = form_data.get("StreamID", "unknown")
    call_uuid = form_data.get("CallUUID", "unknown")

    logger.info(f"🔊 Stream event — Event={event}, StreamID={stream_id}, CallUUID={call_uuid}")
    return Response(content="OK", status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "ngrok_url": NGROK_URL}


# ---------------------------------------------------------------------------
# ngrok Tunnel
# ---------------------------------------------------------------------------

def setup_ngrok():
    """Create ngrok tunnels for HTTP and WebSocket servers."""
    global NGROK_URL

    if NGROK_AUTH_TOKEN:
        conf.get_default().auth_token = NGROK_AUTH_TOKEN

    # Create HTTP tunnel
    http_tunnel = ngrok.connect(HTTP_PORT, "http")
    NGROK_URL = http_tunnel.public_url

    # Ensure HTTPS
    if NGROK_URL.startswith("http://"):
        NGROK_URL = NGROK_URL.replace("http://", "https://")

    logger.info(f"")
    logger.info(f"{'=' * 60}")
    logger.info(f"🌐 ngrok tunnel established!")
    logger.info(f"")
    logger.info(f"   HTTP URL:    {NGROK_URL}")
    logger.info(f"   Answer URL:  {NGROK_URL}/answer")
    logger.info(f"   Hangup URL:  {NGROK_URL}/hangup")
    logger.info(f"")
    logger.info(f"   ➡️  Set the Answer URL in your Vobiz Application settings")
    logger.info(f"   ➡️  Or use make_call.py to trigger an outbound call")
    logger.info(f"{'=' * 60}")
    logger.info(f"")

    return NGROK_URL


# ---------------------------------------------------------------------------
# WebSocket proxy (forward ngrok WSS → local agent WS)
# ---------------------------------------------------------------------------
# Since ngrok tunnels HTTP, we need to handle WebSocket upgrade.
# FastAPI with uvicorn supports WebSocket natively.

from starlette.websockets import WebSocket as StarletteWebSocket
import websockets as ws_lib


@app.websocket("/ws")
async def websocket_proxy(websocket: StarletteWebSocket):
    """
    Proxy WebSocket connection from Vobiz (via ngrok) to the agent server.
    This allows a single ngrok tunnel to handle both HTTP and WebSocket.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted from Vobiz (via ngrok)")

    # Connect to the local agent WebSocket server
    agent_url = f"ws://127.0.0.1:{WS_PORT}"
    try:
        async with ws_lib.connect(agent_url) as agent_ws:
            logger.info(f"Connected to agent at {agent_url}")

            async def forward_to_agent():
                """Forward messages from Vobiz → Agent."""
                try:
                    while True:
                        data = await websocket.receive_text()
                        await agent_ws.send(data)
                except Exception:
                    pass

            async def forward_to_vobiz():
                """Forward messages from Agent → Vobiz."""
                try:
                    async for message in agent_ws:
                        await websocket.send_text(message)
                except Exception:
                    pass

            # Run both directions concurrently
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(forward_to_agent()),
                    asyncio.create_task(forward_to_vobiz()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()

    except Exception as e:
        logger.error(f"Agent WebSocket connection error: {e}")
    finally:
        logger.info("WebSocket proxy connection closed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_agent_server():
    """Start the agent WebSocket server in a separate thread."""
    from agent import start_agent_server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_agent_server())
    loop.run_forever()


def main():
    """Start everything: agent server, ngrok tunnel, HTTP server."""
    logger.info("🚀 Starting Vobiz Voice Agent Server...")

    # 1. Start agent WebSocket server in background thread
    agent_thread = threading.Thread(target=run_agent_server, daemon=True)
    agent_thread.start()
    logger.info(f"✅ Agent WebSocket server starting on port {WS_PORT}")

    # Give the agent server a moment to start
    import time
    time.sleep(1)

    # 2. Setup ngrok tunnel
    try:
        setup_ngrok()
    except Exception as e:
        logger.error(f"❌ Failed to setup ngrok: {e}")
        logger.error("Make sure ngrok is installed: pip install pyngrok")
        logger.error("And authenticated: ngrok authtoken YOUR_TOKEN")
        sys.exit(1)

    # 3. Start HTTP server (blocking)
    logger.info(f"✅ HTTP server starting on port {HTTP_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="info")


if __name__ == "__main__":
    main()
