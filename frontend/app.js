// ==============================================================================
// 1. STATE & DOM ELEMENTS
// ==============================================================================
let latestPayload = null;
let currentSeverityState = "NOMINAL";
let hasReceivedTelemetry = false;
let websocket = null;
let isPaused = false;
let playbackSpeed = 1;
let controlsInitialized = false;
let controlPending = false;

// DOM Elements
const riskScoreDisplay = document.getElementById("risk-score");
const statusIndicator = document.getElementById("status-indicator");
const actionLogContainer = document.getElementById("action-log");
const websocketStatus = document.getElementById("ws-status");
const modeTag = document.getElementById("mode-tag");
const pauseButton = document.getElementById("pause-button");
const speedSelect = document.getElementById("speed-select");
const chartSeries = [
    { key: "ip_current", canvas: "ip-chart", label: "Plasma Current (Ip)", color: "#38bdf8" },
    { key: "electron_temp", canvas: "te-chart", label: "Electron Temperature (Te)", color: "#fbbf24" },
    { key: "magnetic_n1", canvas: "magnetic-chart", label: "Magnetic Fluctuation (n=1)", color: "#c084fc" },
    { key: "density", canvas: "density-chart", label: "Electron Density (Ne)", color: "#34d399" },
    { key: "radiated_power", canvas: "power-chart", label: "Radiated Power (Prad)", color: "#fb7185" },
];
const history = Object.fromEntries(chartSeries.map(series => [series.key, []]));
const historyLimit = 60;

// ==============================================================================
// 2. WEBSOCKET LISTENER (Runs at 50Hz / 20ms)
// ==============================================================================
function connectWebSocket() {
    // Connect to the FastAPI backend
    websocket = new WebSocket("ws://localhost:8000/ws/telemetry");

    websocket.onopen = () => {
        websocketStatus.innerText = "CONNECTED";
        websocketStatus.className = "text-green";
        controlsInitialized = false;
        controlPending = false;
        console.log("[WebSocket] Connected to backend");
    };

    websocket.onmessage = (event) => {
        // Silently update the memory state as fast as packets arrive
        try {
            const payload = JSON.parse(event.data);
            if (payload.type === "error") {
                controlPending = false;
                controlsInitialized = false;
                console.error("[WebSocket] Control error:", payload.message);
                return;
            }
            latestPayload = payload;
            if (
                controlPending
                && payload.paused === isPaused
                && Number(payload.speed) === playbackSpeed
            ) {
                controlPending = false;
            }
        } catch (error) {
            console.error("[WebSocket] Ignoring invalid telemetry payload", error);
        }
    };

    websocket.onclose = () => {
        websocketStatus.innerText = "DISCONNECTED - RECONNECTING";
        websocketStatus.className = "text-muted";
        console.warn("[WebSocket] Disconnected. Reconnecting in 2s...");
        setTimeout(connectWebSocket, 2000);
    };

    websocket.onerror = () => {
        websocketStatus.innerText = "CONNECTION ERROR";
        websocketStatus.className = "text-muted";
    };
}

// ==============================================================================
// 3. UI RENDER LOOP (Throttled to 2Hz / 500ms)
// ==============================================================================
function updateDashboard() {
    if (!latestPayload) return;
    if (!hasReceivedTelemetry) {
        actionLogContainer.replaceChildren();
        hasReceivedTelemetry = true;
    }

    // 1. Update Core Telemetry & Risk Score
    const riskPercentage = (latestPayload.risk * 100).toFixed(1);
    riskScoreDisplay.innerText = `${riskPercentage}%`;
    modeTag.innerText = latestPayload.mode === "demo"
        ? "by team Working Title"
        : "LIVE MODEL OUTPUT";
    pauseButton.disabled = latestPayload.mode !== "demo";
    speedSelect.disabled = latestPayload.mode !== "demo";
    if (
        latestPayload.mode === "demo"
        && (!controlsInitialized || !controlPending)
    ) {
        isPaused = Boolean(latestPayload.paused);
        playbackSpeed = Number(latestPayload.speed) || 1;
        pauseButton.innerText = isPaused ? "Resume" : "Pause";
        speedSelect.value = String(playbackSpeed);
        controlsInitialized = true;
    }

    // 2. Determine Highest Severity in Current Actions
    const actions = latestPayload.actions;
    let highestSeverity = "NOMINAL";

    if (actions.some(a => a.severity === "ABORT")) highestSeverity = "ABORT";
    else if (actions.some(a => a.severity === "CRITICAL")) highestSeverity = "CRITICAL";
    else if (actions.some(a => a.severity === "WARNING")) highestSeverity = "WARNING";

    // 3. Update UI Colors Based on Severity
    updateVisualTheme(highestSeverity);

    // 4. Update Automated Log (Only on state change to prevent spam)
    if (highestSeverity !== currentSeverityState) {
        if (highestSeverity === "NOMINAL" && currentSeverityState !== "NOMINAL") {
            logRecovery(latestPayload.timestamp);
        } else {
            logMitigationActions(actions, highestSeverity, latestPayload.timestamp);
        }
        currentSeverityState = highestSeverity;
    }

    const telemetry = latestPayload.telemetry;
    for (const series of chartSeries) {
        appendHistory(history[series.key], telemetry[series.key]);
        drawTelemetryChart(
            series.canvas,
            history[series.key],
            series.label,
            series.color,
        );
    }
}

// ==============================================================================
// 4. HELPER FUNCTIONS
// ==============================================================================
function appendHistory(values, value) {
    if (!Number.isFinite(value)) return;
    values.push(value);
    if (values.length > historyLimit) values.shift();
}

function drawTelemetryChart(canvasId, values, label, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || values.length === 0) return;

    const context = canvas.getContext("2d");
    const bounds = canvas.getBoundingClientRect();
    const pixelRatio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(bounds.width * pixelRatio));
    canvas.height = Math.max(1, Math.round(bounds.height * pixelRatio));
    context.scale(pixelRatio, pixelRatio);

    const width = bounds.width;
    const height = bounds.height;
    const padding = { top: 28, right: 12, bottom: 12, left: 12 };
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || Math.max(Math.abs(max) * 0.1, 1);
    const plotHeight = height - padding.top - padding.bottom;

    context.clearRect(0, 0, width, height);
    context.fillStyle = "#9ca3af";
    context.font = '12px "MS Courier New", "Courier New", Courier, monospace';
    context.fillText(`${label}: ${values[values.length - 1].toFixed(3)}`, padding.left, 17);

    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    values.forEach((value, index) => {
        const x = padding.left + index * (width - padding.left - padding.right)
            / Math.max(values.length - 1, 1);
        const y = padding.top + (max - value) / range * plotHeight;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
    });
    context.stroke();
}

function updateVisualTheme(severity) {
    // Reset classes
    statusIndicator.className = "status-box";

    switch (severity) {
        case "NOMINAL":
            statusIndicator.classList.add("bg-green", "text-white");
            statusIndicator.innerText = "SYSTEM STABLE";
            break;
        case "WARNING":
            statusIndicator.classList.add("bg-yellow", "text-black");
            statusIndicator.innerText = "TEARING MODE DETECTED";
            break;
        case "CRITICAL":
            statusIndicator.classList.add("bg-red", "text-white", "animate-pulse");
            statusIndicator.innerText = "INSTABILITY CRITICAL";
            break;
        case "ABORT":
            statusIndicator.classList.add("bg-dark-red", "text-white", "animate-flash");
            statusIndicator.innerText = "DISRUPTION IMMINENT - QUENCHING";
            break;
    }
}

function logMitigationActions(actions, severity, unixTimestamp) {
    const timeString = new Date(unixTimestamp * 1000).toISOString().substr(11, 12); // HH:MM:SS.mmm

    // Iterate through all actions triggered by the rules engine
    actions.forEach(action => {
        // Skip logging continuous nominal states
        if (action.action === "MAINTAIN_STATE") return;

        const logEntry = document.createElement("div");
        logEntry.className = `log-entry severity-${severity.toLowerCase()}`;

        logEntry.innerHTML = `
            <span class="log-time">[${timeString}]</span>
            <span class="log-system">${action.system}:</span>
            <span class="log-action">${action.action}</span>
            <div class="log-desc text-muted">${action.description}</div>
        `;

        // Prepend to show newest events at the top
        actionLogContainer.prepend(logEntry);
        while (actionLogContainer.children.length > 100) {
            actionLogContainer.lastElementChild.remove();
        }
    });
}

function logRecovery(unixTimestamp) {
    const logEntry = document.createElement("div");
    logEntry.className = "log-entry severity-nominal";
    const timeString = new Date(unixTimestamp * 1000).toISOString().substr(11, 12);
    logEntry.innerHTML = `
        <span class="log-time">[${timeString}]</span>
        <span class="log-system">SYSTEM:</span>
        <span class="log-action">RETURNED_TO_NOMINAL</span>
        <div class="log-desc text-muted">Risk has returned to the nominal range.</div>
    `;
    actionLogContainer.prepend(logEntry);
    while (actionLogContainer.children.length > 100) {
        actionLogContainer.lastElementChild.remove();
    }
}

function sendPlaybackControl() {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) return;
    controlPending = true;
    websocket.send(JSON.stringify({
        type: "control",
        paused: isPaused,
        speed: playbackSpeed,
    }));
}

pauseButton.addEventListener("click", () => {
    isPaused = !isPaused;
    pauseButton.innerText = isPaused ? "Resume" : "Pause";
    sendPlaybackControl();
});

speedSelect.addEventListener("change", () => {
    playbackSpeed = Number(speedSelect.value);
    sendPlaybackControl();
});

// ==============================================================================
// 5. INITIALIZATION
// ==============================================================================
connectWebSocket();
// Lock the UI rendering to 500ms intervals
setInterval(updateDashboard, 500);