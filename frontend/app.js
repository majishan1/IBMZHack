let latestPayload = null;
let currentSeverityState = null;
let mockMode = false;

// DOM Elements (Safely selecting to avoid crashes if elements are missing from HTML)
const riskScoreDisplay = document.getElementById("risk-score");
const statusIndicator = document.getElementById("status-indicator");
const actionLogContainer = document.getElementById("action-log");

const riskChart = new Chart(document.getElementById('riskChart').getContext('2d'), {
    type: 'line',
    data: { 
        labels: [], 
        datasets: [{ label: 'Disruption Risk', data: [], borderColor: '#e74c3c', borderWidth: 2, tension: 0.4, pointRadius: 0 }] 
    },
    options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { y: { min: 0, max: 1 } } }
});

const telemetryChart = new Chart(document.getElementById('telemetryChart').getContext('2d'), {
    type: 'line',
    data: { 
        labels: [], 
        datasets: [{ label: 'Core Temp (keV)', data: [], borderColor: '#3498db', borderWidth: 2, tension: 0.4, pointRadius: 0 }] 
    },
    options: { responsive: true, maintainAspectRatio: false, animation: false }
});

function connectWebSocket() {
    try {
        const ws = new WebSocket("ws://localhost:8000/ws/telemetry");

        ws.onopen = () => {
            console.log("[WebSocket] Connected");
            mockMode = false;
        };

        ws.onmessage = (event) => {
            latestPayload = JSON.parse(event.data);
        };

        ws.onerror = () => {
            console.warn("[WebSocket] Backend not found. Running local simulation...");
            mockMode = true;
        };

        ws.onclose = () => {
            if (!mockMode) setTimeout(connectWebSocket, 2000);
        };
    } catch (e) {
        mockMode = true;
    }
}

function updateDashboard() {
    if (mockMode) generateMockData();
    if (!latestPayload) return;

    // 1. Update Core Telemetry Text (if elements exist)
    if (riskScoreDisplay) {
        riskScoreDisplay.innerText = `${(latestPayload.risk * 100).toFixed(1)}%`;
    }

    // 2. Determine Highest Severity
    const actions = latestPayload.actions || [];
    let highestSeverity = "NOMINAL";
    if (actions.some(a => a.severity === "ABORT")) highestSeverity = "ABORT";
    else if (actions.some(a => a.severity === "CRITICAL")) highestSeverity = "CRITICAL";
    else if (actions.some(a => a.severity === "WARNING")) highestSeverity = "WARNING";

    // 3. Update Status Indicator (if element exists)
    if (statusIndicator) updateVisualTheme(highestSeverity);

    // 4. Update Automated Log
    if (highestSeverity !== currentSeverityState && actionLogContainer) {
        logMitigationActions(actions, highestSeverity, latestPayload.timestamp);
        currentSeverityState = highestSeverity;
    }

    // 5. Update Charts
    updateCharts(latestPayload);
}

function updateVisualTheme(severity) {
    statusIndicator.className = "status-box";
    switch (severity) {
        case "NOMINAL": statusIndicator.innerText = "SYSTEM STABLE"; break;
        case "WARNING": statusIndicator.innerText = "TEARING MODE DETECTED"; break;
        case "CRITICAL": statusIndicator.innerText = "INSTABILITY CRITICAL"; break;
        case "ABORT": statusIndicator.innerText = "DISRUPTION IMMINENT"; break;
    }
}

function logMitigationActions(actions, severity, unixTimestamp) {
    const timeString = new Date(unixTimestamp * 1000).toISOString().substr(11, 12); 

    actions.forEach(action => {
        if (action.action === "MAINTAIN_STATE") return;

        const logEntry = document.createElement("div");
        logEntry.className = `log-entry severity-${severity.toLowerCase()}`;
        
        logEntry.innerHTML = `
            <span class="log-time">[${timeString}]</span>
            <span class="log-system" style="color: #4da6ff; font-weight:bold;">${action.system}:</span>
            <span class="log-action" style="font-weight:bold;">${action.action}</span>
            <div class="log-desc" style="color: #999;">${action.description}</div>
        `;

        actionLogContainer.prepend(logEntry);
    });
}

function updateCharts(payload) {
    const timeLabel = new Date(payload.timestamp * 1000).toISOString().substr(14, 5); // mm:ss
    
    // Manage Risk Chart
    riskChart.data.labels.push(timeLabel);
    riskChart.data.datasets[0].data.push(payload.risk);
    
    // Manage Telemetry Chart (using mock temp data)
    telemetryChart.data.labels.push(timeLabel);
    telemetryChart.data.datasets[0].data.push(payload.temp || (Math.random() * 5 + 15));

    // Keep arrays at a max of 20 data points
    if (riskChart.data.labels.length > 20) {
        riskChart.data.labels.shift();
        riskChart.data.datasets[0].data.shift();
        telemetryChart.data.labels.shift();
        telemetryChart.data.datasets[0].data.shift();
    }

    riskChart.update();
    telemetryChart.update();
}

// ==============================================================================
// 6. MOCK DATA GENERATOR
// ==============================================================================
let mockRisk = 0.1;
function generateMockData() {
    mockRisk += (Math.random() - 0.45) * 0.15; 
    if (mockRisk < 0.05) mockRisk = 0.05;
    if (mockRisk > 1.2) mockRisk = 0.05; // Drop back to nominal to loop the simulation

    let mockActions = [];
    if (mockRisk > 0.85) mockActions.push({ severity: "ABORT", system: "PELLET_INJECTOR", action: "FIRE_Shattered_Pellet", description: "Injecting Neon/Argon mix to rapidly cool plasma." });
    else if (mockRisk > 0.6) mockActions.push({ severity: "CRITICAL", system: "MAGNET_COILS", action: "ADJUST_POLOIDAL_FIELD", description: "Shaping field to counter severe boundary distortion." });
    else if (mockRisk > 0.3) mockActions.push({ severity: "WARNING", system: "ECRH_HEATING", action: "TARGET_MAGNETIC_ISLAND", description: "Directing microwaves to suppress tearing mode." });
    else mockActions.push({ severity: "NOMINAL", system: "CONTROL_SYSTEM", action: "MAINTAIN_STATE", description: "All parameters nominal." });

    latestPayload = { risk: Math.min(mockRisk, 1), temp: 15 + (mockRisk * 10), timestamp: Date.now() / 1000, actions: mockActions };
}

connectWebSocket();
setInterval(updateDashboard, 500);