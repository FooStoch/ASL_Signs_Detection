# assembly_streaming.py
# This script streams audio from the microphone to AssemblyAI and prints transcripts

import pyaudio
import websocket
import json
import threading
import time
from urllib.parse import urlencode
from datetime import datetime

# --- Configuration ---
YOUR_API_KEY = "ce4ee1242ea544dfb103b7e98e824690"
CONNECTION_PARAMS = {
    "sample_rate": 16000,
    "format_turns": True,
}
API_ENDPOINT = f"wss://streaming.assemblyai.com/v3/ws?{urlencode(CONNECTION_PARAMS)}"
FRAMES_PER_BUFFER = 800
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Globals
audio = None
stream = None
ws_app = None
stop_event = threading.Event()

def on_open(ws):
    def stream_audio():
        global stream
        print("[Streaming] Start sending audio...")
        while not stop_event.is_set():
            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            ws.send(data, websocket.ABNF.OPCODE_BINARY)
        print("[Streaming] Stopped.")

    threading.Thread(target=stream_audio, daemon=True).start()

def on_message(ws, message):
    try:
        data = json.loads(message)
        if data.get("type") == "Turn":
            transcript = data.get("transcript", "")
            if data.get("turn_is_final") and transcript.strip():
                print("Transcript:", transcript)
                with open("transcript_output.txt", "a") as f:
                    f.write(transcript + "\n")
    except Exception as e:
        print("[Error] Message parse failed:", e)

def on_error(ws, error):
    print("[WebSocket Error]", error)
    stop_event.set()

def on_close(ws, code, reason):
    print(f"[WebSocket Closed] {code} - {reason}")
    stop_event.set()
    if stream: stream.close()
    if audio: audio.terminate()

def run():
    global stream, audio, ws_app
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=SAMPLE_RATE, input=True,
                        frames_per_buffer=FRAMES_PER_BUFFER)

    ws_app = websocket.WebSocketApp(
        API_ENDPOINT,
        header={"Authorization": YOUR_API_KEY},
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws_thread = threading.Thread(target=ws_app.run_forever)
    ws_thread.daemon = True
    ws_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
        if ws_app.sock and ws_app.sock.connected:
            ws_app.send(json.dumps({"type": "Terminate"}))
        time.sleep(2)
        ws_app.close()

if __name__ == "__main__":
    run()
