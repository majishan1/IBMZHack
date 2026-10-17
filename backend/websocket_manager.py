import asyncio
from typing import List, Dict, Any
from fastapi import WebSocket


class WebSocketManager:
    """Thread-safe WebSocket manager for high-frequency broadcasting."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Accepts and stores incoming client connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        print(f"[WebSocket] Client connected. Total active clients: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        """Safely removes disconnected clients from pool."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        print(f"[WebSocket] Client disconnected. Total active clients: {len(self.active_connections)}")

    async def broadcast(self, payload: Dict[str, Any]):
        """Broadcasts a JSON payload concurrently to all connected dashboard instances.

        Args:
            payload: Python dictionary representing data packet
        """
        if not self.active_connections:
            return

        async with self._lock:
            # Create snapshot of connections to safely modify list during loop
            connections = list(self.active_connections)

        disconnected_clients = []
        
        # Dispatch JSON payloads across active sockets concurrently
        for websocket in connections:
            try:
                await websocket.send_json(payload)
            except Exception:
                # Flag client for disconnection on socket write error
                disconnected_clients.append(websocket)

        # Clean up stale connections
        for stale_client in disconnected_clients:
            await self.disconnect(stale_client)

    @property
    def client_count(self) -> int:
        """Returns active client count."""
        return len(self.active_connections)


# Singleton instance for import throughout backend
websocket_manager = WebSocketManager()