// ==============================================================================
// 1. STATE & DOM ELEMENTS
// ==============================================================================
let latestPayload = null;
let currentSeverityState = "NOMINAL";

// DOM Elements
const riskScoreDisplay = document.getElementById("risk-score");
const statusIndicator = document.getElementById("status-indicator");
const actionLogContainer = document.getElementById("action-log");

// ==============================================================================
// 2. WEBSOCKET LISTENER (Runs at 50Hz / 20ms)
// ==============================================================================
function connectWebSocket() {
    // Connect to the FastAPI backend
    const ws = new WebSocket("ws://localhost:8000/ws/telemetry");

    ws.onopen = () => console.log("[WebSocket] Connected to IBM Z Backend");

    ws.onmessage = (event) => {
        // Silently update the memory state as fast as packets arrive
        latestPayload = JSON.parse(event.data);
    };

    ws.onclose = () => {
        console.warn("[WebSocket] Disconnected. Reconnecting in 2s...");
        setTimeout(connectWebSocket, 2000);
    };
}

// ==============================================================================
// 3. UI RENDER LOOP (Throttled to 2Hz / 500ms)
// ==============================================================================
function updateDashboard() {
    if (!latestPayload) return;

    // 1. Update Core Telemetry & Risk Score
    const riskPercentage = (latestPayload.risk * 100).toFixed(1);
    riskScoreDisplay.innerText = `${riskPercentage}%`;

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
        logMitigationActions(actions, highestSeverity, latestPayload.timestamp);
        currentSeverityState = highestSeverity;
    }

    // (Optional) Update Chart.js instances here using latestPayload.telemetry
}

// ==============================================================================
// 4. HELPER FUNCTIONS
// ==============================================================================
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
    });
}

// ==============================================================================
// 5. INITIALIZATION
// ==============================================================================
connectWebSocket();
// Lock the UI rendering to 500ms intervals
setInterval(updateDashboard, 500);