#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo "  STARTING TOKAMAK PLASMA DISRUPTION PREDICTOR SYSTEMS   "
echo "  IBM Z LinuxONE Accelerator Environment                 "
echo "=========================================================="

# 1. HARDWARE & THREAD CONFIGURATION
export ZAIUDEVICES=1
export OMP_NUM_THREADS=4
echo "[+] Hardware acceleration enabled: ZAIUDEVICES=1"
echo "[+] PyTorch OMP threads locked to: 4"

# 2. CHECK FOR TRAINED MODEL
if [ ! -f "checkpoints/compiled_lstm.pt" ]; then
    echo "[-] WARNING: 'checkpoints/compiled_lstm.pt' not found!"
    echo "[!] Running model/train.py first to compile TorchScript weights..."
    python3 model/train.py
fi

# 3. CLEAN PROCESS TEARDOWN ON EXIT
# Ensures background jobs (backend & frontend) are killed when you press Ctrl+C
cleanup() {
    echo -e "\n[!] Shutting down all processes..."
    kill $(jobs -p) 2>/dev/null || true
    echo "[+] Cleanup complete. Goodbye!"
    exit 0
}
trap cleanup SIGINT SIGTERM

# 4. START FASTAPI BACKEND (Port 8000)
echo "[+] Launching FastAPI backend server on http://localhost:8000..."
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait briefly to ensure backend initializes before launching UI
sleep 2

# 5. START FRONTEND HTTP SERVER (Port 3000)
echo "[+] Launching Frontend Dashboard on http://localhost:3000..."
python3 -m http.server 3000 --directory frontend &
FRONTEND_PID=$!

echo "=========================================================="
echo "  SYSTEM RUNNING!                                         "
echo "  - Backend API & WebSockets: http://localhost:8000       "
echo "  - Operator Dashboard:       http://localhost:3000       "
echo "  Press [CTRL+C] to stop all servers cleanly.            "
echo "=========================================================="

# Keep script running to listen for Ctrl+C
wait