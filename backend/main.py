import asyncio
import math
import os
import random
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

DEMO_MODE = os.getenv("TOKAMAK_DEMO_MODE", "1").lower() not in {
    "0",
    "false",
    "no",
}

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
# 3. TELEMETRY LOOPS (50Hz / 20ms Loop)
# ==============================================================================
async def plasma_monitoring_loop():
    """Publish live model output, or clearly marked synthetic demo telemetry."""
    from logic.mitigation_rules import evaluate_mitigation

    if DEMO_MODE:
        await demo_monitoring_loop(evaluate_mitigation)
        return

    import torch
    from data_pipeline.normalizer import SensorNormalizer
    from data_pipeline.stream_client import RemoteTokamakStreamer
    from model.inference import engine as inference_engine

    torch.set_num_threads(4)
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

            suggested_actions = evaluate_mitigation(risk_score)

            payload = {
                "timestamp": time.time(),
                "risk": float(risk_score),
                "mode": "live",
                "actions": suggested_actions,
                "telemetry": {
                    "ip_current": float(raw_sensor_batch[0, -1, 0]),
                    "electron_temp": float(raw_sensor_batch[0, -1, 1]),
                    "magnetic_n1": float(raw_sensor_batch[0, -1, 2]),
                    "density": float(raw_sensor_batch[0, -1, 3]),
                    "radiated_power": float(raw_sensor_batch[0, -1, 4]),
                },
            }

            await manager.broadcast(payload)

        except Exception as e:
            # Log non-fatal streaming error and maintain loop frequency
            print(f"[Loop Exception] Error in pipeline processing: {str(e)}")

        # Precise 20ms sleep drift compensation
        elapsed_time = time.perf_counter() - loop_start_time
        sleep_duration = max(0.0, 0.02 - elapsed_time)
        await asyncio.sleep(sleep_duration)


async def demo_monitoring_loop(evaluate_mitigation):
    """Stream randomized, bounded synthetic diagnostics for demonstration only."""
    simulator = SyntheticTokamak()
    while True:
        telemetry, risk_score = simulator.sample()
        payload = {
            "timestamp": time.time(),
            "risk": risk_score,
            "mode": "demo",
            "actions": evaluate_mitigation(risk_score),
            "telemetry": telemetry,
        }
        await manager.broadcast(payload)
        await asyncio.sleep(0.02)


class SyntheticTokamak:
    """Generate plausible-looking diagnostics with slowly changing plasma state."""

    def __init__(self):
        self.random = random.Random()
        self.state = "nominal"
        self.state_age = 0
        self.step = 0
        self.risk = 0.12
        self.signals = {
            "ip_current": 0.95,
            "electron_temp": 3.2,
            "magnetic_n1": 0.0,
            "density": 1.8,
            "radiated_power": 1.2,
        }

    def _advance_state(self):
        self.state_age += 1
        if self.state_age < 250:
            return

        transition_roll = self.random.random()
        if self.state == "nominal" and transition_roll < 0.0007:
            self.state = "tearing"
            self.state_age = 0
        elif self.state == "tearing":
            if transition_roll < 0.002:
                self.state = "critical"
                self.state_age = 0
            elif transition_roll < 0.012:
                self.state = "nominal"
                self.state_age = 0
        elif self.state == "critical" and transition_roll < 0.01:
            self.state = "recovery"
            self.state_age = 0
        elif self.state == "recovery" and transition_roll < 0.01:
            self.state = "nominal"
            self.state_age = 0

    def sample(self):
        self.step += 1
        self._advance_state()
        severity = {"nominal": 0, "recovery": 0, "tearing": 1, "critical": 2}[
            self.state
        ]
        oscillation = math.sin(self.step * 0.035)
        targets = {
            "ip_current": 0.95 - 0.045 * severity,
            "electron_temp": 3.2 - 0.48 * severity,
            "magnetic_n1": 0.002 + 0.012 * severity,
            "density": 1.8 + 0.23 * severity,
            "radiated_power": 1.2 + 1.15 * severity,
        }
        amplitudes = {
            "ip_current": 0.0018,
            "electron_temp": 0.045,
            "magnetic_n1": 0.0012 + 0.002 * severity,
            "density": 0.018,
            "radiated_power": 0.04 + 0.035 * severity,
        }
        bounds = {
            "ip_current": (0.0, 2.5),
            "electron_temp": (0.1, 10.0),
            "magnetic_n1": (-0.05, 0.05),
            "density": (0.1, 5.0),
            "radiated_power": (0.0, 8.0),
        }

        for name, target in targets.items():
            minimum, maximum = bounds[name]
            previous = self.signals[name]
            noise = self.random.gauss(0.0, amplitudes[name])
            self.signals[name] = min(
                maximum,
                max(minimum, previous + 0.025 * (target - previous) + noise),
            )

        target_risk = (0.12, 0.52, 0.82)[severity]
        risk_noise = self.random.gauss(0.0, 0.008)
        self.risk = min(0.99, max(0.01, self.risk + 0.04 * (target_risk - self.risk) + risk_noise))

        telemetry = dict(self.signals)
        telemetry["magnetic_n1"] += oscillation * (0.0006 + severity * 0.001)
        telemetry["magnetic_n1"] = min(
            bounds["magnetic_n1"][1],
            max(bounds["magnetic_n1"][0], telemetry["magnetic_n1"]),
        )
        return telemetry, self.risk


# ==============================================================================
# 4. LIFECYCLE & ROUTES
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    """Start the selected telemetry stream on application boot."""
    asyncio.create_task(plasma_monitoring_loop())


@app.get("/health")
async def health_check():
    """System health verification endpoint."""
    return {
        "status": "online",
        "mode": "demo" if DEMO_MODE else "live",
        "hardware": "synthetic demo data" if DEMO_MODE else "CPU inference",
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