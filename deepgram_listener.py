import os
import json
import queue
import threading
import time
import requests
import sounddevice as sd
import websocket
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

audio_queue = queue.Queue()
ws_open = False


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)

    audio_queue.put(indata.copy().tobytes())


def analyze_text(text):
    try:
        response = requests.post(
            "http://127.0.0.1:8081/detect",
            json={"text": text},
            timeout=10,
        )

        return response.json()

    except Exception as e:
        return {"error": str(e)}


def main():

    global ws_open

    if not DEEPGRAM_API_KEY:
        print("Falta DEEPGRAM_API_KEY en .env")
        return

    deepgram_url = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-3"
        "&language=es"
        "&encoding=linear16"
        "&sample_rate=16000"
        "&channels=1"
        "&interim_results=true"
        "&punctuate=true"
    )

    def on_open(ws):
        global ws_open
        ws_open = True

        print("✅ Conectado a Deepgram")
        print("🎤 Habla al micrófono...\n")

    def on_message(ws, message):

        try:

            data = json.loads(message)

            if "channel" in data and data["channel"]["alternatives"]:

                alt = data["channel"]["alternatives"][0]

                transcript = alt.get("transcript", "")
                is_final = data.get("is_final", False)

                # ignorar resultados intermedios
                if not is_final:
                    return

                if transcript.strip():

                    print(f"\n📝 Texto: {transcript}")

                    result = analyze_text(transcript)

                    if "error" in result:
                        print("❌ Error analizando texto:", result["error"])

                    else:

                        risk = result["violence_risk"]

                        if risk == "HIGH":
                            print("🚨 ALERTA: POSIBLE VIOLENCIA VERBAL DETECTADA")

                        elif risk == "MEDIUM":
                            print("⚠ Lenguaje agresivo detectado")

                        else:
                            print("🟢 Conversación normal")

                        print(
                            f"Riesgo: {risk} | "
                            f"Sentimiento: {result['sentiment']} | "
                            f"Confianza: {result['confidence']:.2f}"
                        )

        except Exception as e:

            print("❌ Error procesando mensaje:", e)

    def on_error(ws, error):
        print("❌ Error Deepgram:", error)

    def on_close(ws, close_status_code, close_msg):

        global ws_open
        ws_open = False

        print(f"🔌 Conexión cerrada: {close_status_code} | {close_msg}")

    ws = websocket.WebSocketApp(
        deepgram_url,
        header=[f"Authorization: Token {DEEPGRAM_API_KEY}"],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()

    print("⏳ Esperando conexión con Deepgram...")

    timeout = 10
    start_time = time.time()

    while not ws_open and (time.time() - start_time) < timeout:
        time.sleep(0.1)

    if not ws_open:
        print("❌ No se pudo abrir la conexión con Deepgram.")
        return

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=audio_callback,
        blocksize=CHUNK_SIZE,
    )

    try:

        with stream:

            while True:

                audio_data = audio_queue.get()

                if ws_open and ws.sock and ws.sock.connected:
                    ws.send(audio_data, opcode=websocket.ABNF.OPCODE_BINARY)

                else:
                    print("⚠ Socket cerrado. Ya no se enviará audio.")
                    break

    except KeyboardInterrupt:

        print("\n🛑 Detenido por usuario.")

    except Exception as e:

        print("❌ Error en captura/envío de audio:", e)

    finally:

        try:
            ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()