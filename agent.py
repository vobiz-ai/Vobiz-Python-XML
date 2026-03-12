"""
agent.py — WebSocket Voice Agent
=================================
Handles bidirectional audio streaming with Vobiz.
Pipeline: Vobiz Audio → Deepgram STT → OpenAI LLM → OpenAI TTS → Vobiz playAudio
"""

import os
import json
import base64
import asyncio
import logging
import struct

import websockets
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
AGENT_SYSTEM_PROMPT = os.getenv(
    "AGENT_SYSTEM_PROMPT",
    "You are a helpful AI phone assistant. Be concise and conversational. Keep responses under 2 sentences.",
)

WS_PORT = int(os.getenv("AGENT_WS_PORT", "5001"))

# Audio settings for Vobiz (mulaw 8kHz)
VOBIZ_SAMPLE_RATE = 8000
VOBIZ_CONTENT_TYPE = "audio/x-mulaw"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agent")

# ---------------------------------------------------------------------------
# OpenAI client (used for both LLM and TTS)
# ---------------------------------------------------------------------------
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Audio conversion helpers
# ---------------------------------------------------------------------------

def _linear_to_mulaw(sample: int) -> int:
    """Convert a 16-bit signed PCM sample to 8-bit μ-law."""
    MULAW_MAX = 0x1FFF
    MULAW_BIAS = 33
    sign = 0
    if sample < 0:
        sign = 0x80
        sample = -sample
    sample = min(sample + MULAW_BIAS, MULAW_MAX)
    exponent = 7
    for exp_val in [0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100]:
        if sample >= exp_val:
            break
        exponent -= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    mulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return mulaw_byte


def pcm16_to_mulaw(pcm_data: bytes) -> bytes:
    """Convert raw 16-bit signed LE PCM to μ-law bytes."""
    samples = struct.unpack(f"<{len(pcm_data) // 2}h", pcm_data)
    return bytes(_linear_to_mulaw(s) for s in samples)


def resample_linear(pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
    """Simple linear resampling of 16-bit PCM data."""
    if from_rate == to_rate:
        return pcm_data
    samples = struct.unpack(f"<{len(pcm_data) // 2}h", pcm_data)
    ratio = from_rate / to_rate
    new_length = int(len(samples) / ratio)
    resampled = []
    for i in range(new_length):
        src_idx = i * ratio
        idx = int(src_idx)
        frac = src_idx - idx
        if idx + 1 < len(samples):
            val = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
        else:
            val = samples[idx]
        resampled.append(max(-32768, min(32767, val)))
    return struct.pack(f"<{len(resampled)}h", *resampled)


# ---------------------------------------------------------------------------
# OpenAI TTS — generate speech audio from text
# ---------------------------------------------------------------------------

async def generate_tts_audio(text: str) -> bytes:
    """Generate speech audio from text using OpenAI TTS. Returns raw mulaw 8kHz bytes."""
    logger.info(f"TTS generating for: {text[:80]}...")

    try:
        response = await openai_client.audio.speech.create(
            model="tts-1",
            voice=OPENAI_TTS_VOICE,
            input=text,
            response_format="pcm",  # Raw 16-bit PCM at 24kHz
        )

        pcm_24k = response.read()

        if not pcm_24k:
            logger.warning("OpenAI TTS returned empty audio")
            return b""

        # Resample from 24kHz to 8kHz
        pcm_8k = resample_linear(pcm_24k, 24000, 8000)

        # Convert to μ-law for Vobiz
        mulaw_data = pcm16_to_mulaw(pcm_8k)

        logger.info(f"TTS audio generated: {len(mulaw_data)} bytes of mulaw")
        return mulaw_data

    except Exception as e:
        logger.error(f"OpenAI TTS error: {e}")
        return b""


# ---------------------------------------------------------------------------
# OpenAI LLM — generate response
# ---------------------------------------------------------------------------

async def get_llm_response(conversation_history: list[dict]) -> str:
    """Get a response from OpenAI given conversation history."""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history,
            max_tokens=150,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        logger.info(f"LLM response: {reply[:80]}...")
        return reply
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return "I'm sorry, I'm having trouble processing that. Could you repeat?"


# ---------------------------------------------------------------------------
# Deepgram STT via raw WebSocket (no SDK dependency issues)
# ---------------------------------------------------------------------------

DEEPGRAM_WS_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2"
    "&language=en"
    "&encoding=mulaw"
    "&sample_rate=8000"
    "&channels=1"
    "&interim_results=true"
    "&utterance_end_ms=1000"
    "&vad_events=true"
    "&endpointing=300"
)


# ---------------------------------------------------------------------------
# Session — per-call state
# ---------------------------------------------------------------------------

class CallSession:
    """Manages state for a single phone call."""

    def __init__(self, ws):
        self.ws = ws  # Vobiz WebSocket connection
        self.stream_id: str | None = None
        self.call_id: str | None = None
        self.is_playing = False
        self.conversation_history: list[dict] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]
        self.transcript_buffer = ""
        self.silence_timer: asyncio.Task | None = None
        self.deepgram_ws = None
        self._deepgram_task: asyncio.Task | None = None

    async def start_deepgram(self):
        """Connect to Deepgram via raw WebSocket for live transcription."""
        try:
            extra_headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
            self.deepgram_ws = await websockets.connect(
                DEEPGRAM_WS_URL,
                additional_headers=extra_headers,
            )
            logger.info("Deepgram STT WebSocket connected")

            # Start listening for transcripts in background
            self._deepgram_task = asyncio.create_task(self._listen_deepgram())
            return True

        except Exception as e:
            logger.error(f"Deepgram connection error: {e}")
            return False

    async def _listen_deepgram(self):
        """Listen for transcript results from Deepgram WebSocket."""
        try:
            async for message in self.deepgram_ws:
                data = json.loads(message)
                msg_type = data.get("type", "")

                if msg_type == "Results":
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    if alternatives:
                        transcript = alternatives[0].get("transcript", "")
                        is_final = data.get("is_final", False)

                        if is_final and transcript.strip():
                            self.transcript_buffer += " " + transcript.strip()
                            logger.info(f"[STT Final] {transcript.strip()}")

                            # Cancel previous silence timer
                            if self.silence_timer and not self.silence_timer.done():
                                self.silence_timer.cancel()

                            # Process after 1.2s of silence
                            self.silence_timer = asyncio.create_task(
                                self._process_after_silence()
                            )

                        elif not is_final and transcript.strip():
                            logger.debug(f"[STT Interim] {transcript.strip()}")

                elif msg_type == "UtteranceEnd":
                    # Deepgram detected end of utterance
                    if self.transcript_buffer.strip():
                        if self.silence_timer and not self.silence_timer.done():
                            self.silence_timer.cancel()
                        self.silence_timer = asyncio.create_task(
                            self._process_after_silence()
                        )

        except websockets.exceptions.ConnectionClosed:
            logger.info("Deepgram WebSocket closed")
        except Exception as e:
            logger.error(f"Deepgram listener error: {e}")

    async def send_audio_to_deepgram(self, audio_bytes: bytes):
        """Send raw audio bytes to Deepgram for transcription."""
        if self.deepgram_ws:
            try:
                await self.deepgram_ws.send(audio_bytes)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Deepgram WebSocket already closed")
                self.deepgram_ws = None
            except Exception as e:
                logger.error(f"Error sending to Deepgram: {e}")

    async def _process_after_silence(self):
        """Wait for silence then process the accumulated transcript."""
        try:
            await asyncio.sleep(1.2)

            user_text = self.transcript_buffer.strip()
            self.transcript_buffer = ""

            if not user_text:
                return

            logger.info(f"Processing user input: {user_text}")

            # Barge-in: interrupt if agent is currently playing
            if self.is_playing:
                await self._clear_audio()

            # Add user message to conversation
            self.conversation_history.append({"role": "user", "content": user_text})

            # Get LLM response
            response_text = await get_llm_response(self.conversation_history)
            self.conversation_history.append({"role": "assistant", "content": response_text})

            # Generate TTS and play back
            audio_data = await generate_tts_audio(response_text)
            if audio_data:
                await self._play_audio(audio_data)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Process after silence error: {e}")

    async def _play_audio(self, mulaw_data: bytes):
        """Send audio to Vobiz via playAudio events in chunks."""
        self.is_playing = True
        chunk_size = 160  # 20ms at 8kHz mono mulaw

        try:
            for i in range(0, len(mulaw_data), chunk_size):
                chunk = mulaw_data[i:i + chunk_size]
                payload = base64.b64encode(chunk).decode("utf-8")

                play_event = {
                    "event": "playAudio",
                    "media": {
                        "contentType": VOBIZ_CONTENT_TYPE,
                        "sampleRate": VOBIZ_SAMPLE_RATE,
                        "payload": payload,
                    },
                }
                await self.ws.send(json.dumps(play_event))

            # Send checkpoint after all audio chunks
            if self.stream_id:
                checkpoint_event = {
                    "event": "checkpoint",
                    "streamId": self.stream_id,
                    "name": f"response-{len(self.conversation_history)}",
                }
                await self.ws.send(json.dumps(checkpoint_event))

            logger.info(f"Sent {len(mulaw_data)} bytes of audio in chunks")

        except Exception as e:
            logger.error(f"Play audio error: {e}")
            self.is_playing = False

    async def _clear_audio(self):
        """Send clearAudio to interrupt playback (barge-in)."""
        if self.stream_id:
            clear_event = {
                "event": "clearAudio",
                "streamId": self.stream_id,
            }
            await self.ws.send(json.dumps(clear_event))
            self.is_playing = False
            logger.info("Sent clearAudio (barge-in)")

    async def handle_message(self, message: str):
        """Process an incoming WebSocket message from Vobiz."""
        try:
            data = json.loads(message)
            event = data.get("event")

            if event == "start":
                self.stream_id = data.get("streamId")
                self.call_id = data.get("callId")
                logger.info(f"Stream started — streamId={self.stream_id}, callId={self.call_id}")

                # Start Deepgram STT
                await self.start_deepgram()

                # Play greeting
                greeting = "Hello! This is the Vobiz AI assistant. How can I help you today?"
                self.conversation_history.append({"role": "assistant", "content": greeting})
                audio_data = await generate_tts_audio(greeting)
                if audio_data:
                    await self._play_audio(audio_data)

            elif event == "media":
                # Forward audio to Deepgram for transcription
                media = data.get("media", {})
                payload = media.get("payload", "")
                if payload:
                    audio_bytes = base64.b64decode(payload)
                    await self.send_audio_to_deepgram(audio_bytes)

            elif event == "playedStream":
                name = data.get("name", "")
                logger.info(f"Checkpoint reached: {name}")
                self.is_playing = False

            elif event == "clearedAudio":
                logger.info("Audio cleared by Vobiz")
                self.is_playing = False

            elif event == "stop":
                logger.info(f"Stream stopped — streamId={self.stream_id}")
                await self.cleanup()

        except json.JSONDecodeError:
            logger.error("Received invalid JSON from Vobiz")
        except Exception as e:
            logger.error(f"Message handler error: {e}")

    async def cleanup(self):
        """Clean up resources when call ends."""
        if self.deepgram_ws:
            try:
                await self.deepgram_ws.close()
            except Exception:
                pass
        if self._deepgram_task and not self._deepgram_task.done():
            self._deepgram_task.cancel()
        if self.silence_timer and not self.silence_timer.done():
            self.silence_timer.cancel()
        logger.info("Session cleaned up")


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

async def handle_connection(websocket, path=None):
    """Handle a new WebSocket connection from Vobiz."""
    logger.info("New WebSocket connection from Vobiz")
    session = CallSession(websocket)

    try:
        async for message in websocket:
            await session.handle_message(message)
    except websockets.exceptions.ConnectionClosed:
        logger.info("Vobiz WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        await session.cleanup()


async def start_agent_server():
    """Start the WebSocket server for the agent."""
    server = await websockets.serve(
        handle_connection,
        "0.0.0.0",
        WS_PORT,
        ping_interval=20,
        ping_timeout=20,
    )
    logger.info(f"🤖 Agent WebSocket server running on ws://0.0.0.0:{WS_PORT}")
    return server


if __name__ == "__main__":
    async def main():
        server = await start_agent_server()
        await asyncio.Future()

    asyncio.run(main())
