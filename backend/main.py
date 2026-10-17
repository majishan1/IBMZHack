import asyncio
import time
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Import local pipeline modules
from data_pipeline.stream_client import RemoteTokamakStreamer
from data_pipeline.normalizer import SensorNormalizer
from model.inference import engine as inference_engine
from logic.mitigation_rules import evaluate_mitigation

# ==============================================================================
# 1. CPU THREAD CONFIGURATION
# ==============================================================================
# Bound CPU thread allocation to reduce contention with the asynchronous server.
torch.set_num_threads(4)

app = FastAPI(title="IBM Z Plasma Disruption Predictor", version="1.0.0")

# Enable CORS for local/remote dashboard connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 2. WEBSOCKET CONNECTION MANAGER
# ==============================================================================
class ConnectionManager:
    """Manages active UI client connections and broadcasts broadcast packages asynchronously."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return

        # Parallelize client frame dispatching
        disconnected_clients = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected_clients.append(connection)

        # Cleanup dropped connections
        for client in disconnected_clients:
            self.disconnect(client)


manager = ConnectionManager()


# ==============================================================================
# 3. CONTINUOUS REAL-TIME INFERENCE LOOP (50Hz / 20ms Loop)
# ==============================================================================
async def plasma_monitoring_loop():
    """Asynchronous background worker thread managing zero-copy 20ms inference loops."""

    # Initialize streamed data pipeline and zero-copy normalizer
    streamer = RemoteTokamakStreamer()
    normalizer = SensorNormalizer()

    while True:
        loop_start_time = time.perf_counter()

        try:
            # 1. Fetch next 20ms slice (1, 10, 5) directly into RAM from streamer
            raw_sensor_batch = streamer.fetch_next_20ms_window()

            # 2. In-place conversion to normalized PyTorch tensor
            normalized_tensor = normalizer.process_raw_batch(raw_sensor_batch)

            # 3. Sub-millisecond TorchScript inference pass on IBM Z
            risk_score = inference_engine.predict_disruption_risk(normalized_tensor)

            # 4. Evaluate automated heuristic action triggers
            suggested_actions = evaluate_mitigation(risk_score)

            # 5. Build compact telemetry payload
            payload = {
                "timestamp": time.time(),
                "risk": float(risk_score),
                "actions": suggested_actions,
                "telemetry": {
                    "ip_current": float(raw_sensor_batch[0, -1, 0]),
                    "electron_temp": float(raw_sensor_batch[0, -1, 1]),
                    "magnetic_n1": float(raw_sensor_batch[0, -1, 2]),
                    "density": float(raw_sensor_batch[0, -1, 3]),
                    "radiated_power": float(raw_sensor_batch[0, -1, 4]),
                },
            }

            # 6. Broadcast packet to WebSocket clients
            await manager.broadcast(payload)

        except Exception as e:
            # Log non-fatal streaming error and maintain loop frequency
            print(f"[Loop Exception] Error in pipeline processing: {str(e)}")

        # Precise 20ms sleep drift compensation
        elapsed_time = time.perf_counter() - loop_start_time
        sleep_duration = max(0.0, 0.02 - elapsed_time)
        await asyncio.sleep(sleep_duration)


# ==============================================================================
# 4. LIFECYCLE & ROUTES
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    """Spawns background continuous 20ms AI evaluation task on application boot."""
    asyncio.create_task(plasma_monitoring_loop())


@app.get("/health")
async def health_check():
    """System health verification endpoint."""
    return {
        "status": "online",
        "accelerator": "IBM Z Telum Enabled (ZAIUDEVICES=1)",
        "active_clients": len(manager.active_connections),
    }


@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time telemetry WebSocket streaming endpoint."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection open and yield control to loop
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
    )