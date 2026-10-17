const commonOptions = { 
    responsive: true, 
    maintainAspectRatio: false, 
    animation: false, 
    scales: { x: { display: false } } 
};

const riskChart = new Chart(document.getElementById('riskChart'), {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'AI Risk Score', borderColor: '#ff4757', backgroundColor: 'rgba(255, 71, 87, 0.1)', fill: true, data: [] }] },
    options: { ...commonOptions, scales: { y: { min: 0, max: 1 } } }
});

const telemetryChart = new Chart(document.getElementById('telemetryChart'), {
    type: 'line',
    data: { 
        labels: [], 
        datasets: [
            { label: 'Ip Current', borderColor: '#1e90ff', data: [] },
            { label: 'Electron Temp', borderColor: '#9b59b6', data: [] }
        ] 
    },
    options: commonOptions
});

let latestData = null;
let lastActionStr = "";
const ws = new WebSocket("ws://localhost:8000/ws/telemetry");

ws.onmessage = (event) => { latestData = JSON.parse(event.data); };

setInterval(() => {
    if (!latestData) return;

    const time = new Date(latestData.timestamp * 1000).toLocaleTimeString();
    const MAX_POINTS = 50;

    riskChart.data.labels.push(time);
    riskChart.data.datasets[0].data.push(latestData.risk);
    
    telemetryChart.data.labels.push(time);
    telemetryChart.data.datasets[0].data.push(latestData.telemetry.ip_current);
    telemetryChart.data.datasets[1].data.push(latestData.telemetry.electron_temp);
    
    if (riskChart.data.labels.length > MAX_POINTS) {
        riskChart.data.labels.shift();
        riskChart.data.datasets[0].data.shift();
        
        telemetryChart.data.labels.shift();
        telemetryChart.data.datasets[0].data.shift();
        telemetryChart.data.datasets[1].data.shift();
    }
    
    riskChart.update();
    telemetryChart.update();

    // assume actions array
    if (latestData.actions && latestData.actions.length > 0) {
        const currentActionStr = JSON.stringify(latestData.actions);
        
        if (currentActionStr !== lastActionStr) {
            const logTerminal = document.getElementById("action-log");
            
            latestData.actions.forEach(action => {
                const entry = document.createElement("div");
                entry.className = `log-entry ${action.severity.toLowerCase()}`;
                entry.innerHTML = `
                    <span class="log-time">[${time}]</span> <strong>[${action.system}]</strong> 
                    <span class="log-action">> ${action.description}</span>
                `;
                logTerminal.appendChild(entry);
            });
            
            logTerminal.scrollTop = logTerminal.scrollHeight;
            lastActionStr = currentActionStr;
        }
    }
    
}, 500);