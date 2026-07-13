// 1. Tab Navigation Routing Logic
document.addEventListener("DOMContentLoaded", () => {
    setupTabs();
    setupSniffer();
    setupSimulator();
    setupXaiViewer();
    setupCharts();
    setupReporting();
});

function setupTabs() {
    const tabs = document.querySelectorAll(".tab-btn");
    const contents = document.querySelectorAll(".tab-content");

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            contents.forEach(c => c.classList.remove("active"));

            tab.classList.add("active");
            const target = tab.getAttribute("data-tab");
            document.getElementById(target).classList.add("active");
        });
    });
}

// 2. Global State & Constant Data mapping
const ATTACK_TYPES = {
    benign: { name: "BENIGN", desc: "Standard, clean network operation.", severity: "LOW", action: "Ignore", scoreRange: [0.002, 0.025], confidenceRange: [98.5, 99.9] },
    ddos: { name: "DDOS FLOOD", desc: "High-density volumetric packet flood targeting local gateways.", severity: "HIGH", action: "Block Source IP", scoreRange: [0.182, 0.384], confidenceRange: [99.2, 99.9] },
    dos: { name: "DOS HULK", desc: "Application layer denial of service attempt exhausting CPU buffers.", severity: "HIGH", action: "Block Source IP", scoreRange: [0.125, 0.282], confidenceRange: [98.1, 99.7] },
    bot: { name: "BOTNET COMMAND", desc: "Active connection attempt to external command-and-control server.", severity: "MEDIUM", action: "Quarantine Device", scoreRange: [0.065, 0.145], confidenceRange: [92.4, 98.8] },
    infiltration: { name: "INFILTRATION PROBE", desc: "Internal host attempting privilege escalation or shell execution.", severity: "HIGH", action: "Quarantine Device", scoreRange: [0.095, 0.215], confidenceRange: [88.5, 97.4] },
    portscan: { name: "SYN PORTSCAN", desc: "Rapid probing of sequential ports to locate service vulnerabilities.", severity: "MEDIUM", action: "Restrict Port", scoreRange: [0.054, 0.115], confidenceRange: [99.5, 99.9] },
    bruteforce: { name: "SSH BRUTE FORCE", desc: "High-frequency dictionary password cracking attempts detected.", severity: "MEDIUM", action: "Block Source IP", scoreRange: [0.058, 0.098], confidenceRange: [97.8, 99.5] },
    webattack: { name: "WEB SQL INJECTION", desc: "Malicious payload input seeking local database command execution.", severity: "MEDIUM", action: "Restrict Port", scoreRange: [0.062, 0.122], confidenceRange: [91.2, 98.4] }
};

const FEATURE_DATA = {
    "destination_port": { desc: "Target port of connection. High attributions in PortScans (Port 80/22/443 mapping vs random).", captum: 0.1824, shap: 0.1542 },
    "flow_duration": { desc: "Total time elapsed in the session flow in microseconds. Volumetric attacks generate outliers.", captum: -0.0452, shap: -0.0384 },
    "flow_packets_s": { desc: "Number of packets transmitted per second. Primary indicator for DDoS flooding vectors.", captum: 0.2842, shap: 0.2612 },
    "fwd_pkt_len_max": { desc: "Maximum length of forward packets. Aids in web shell and exploit classification.", captum: 0.1105, shap: 0.0985 },
    "bwd_pkt_len_min": { desc: "Minimum length of backward packets. Helps verify standard TCP handshakes.", captum: -0.0824, shap: -0.0712 },
    "flow_iat_mean": { desc: "Mean inter-arrival time between packets. Small intervals reveal autonomous bot commands.", captum: -0.1584, shap: -0.1345 },
    "active_mean": { desc: "Mean active time of connection flow before going idle.", captum: 0.0321, shap: 0.0284 },
    "idle_mean": { desc: "Mean time connection spent in sleep mode.", captum: -0.0152, shap: -0.0115 }
};

// Threat logs tracking
let totalPackets = 0;
let threatsCount = 0;
let anomalySum = 0;
let threatLog = [];
let isSniffing = false;
let snifferInterval = null;

// Charts instances
let anomalyChart = null;
let threatDistChart = null;
let globalImportanceChart = null;

const anomalyDataPoints = Array(20).fill(0.015);
const anomalyLabels = Array(20).fill("");
let chartStep = 0;

const threatCounts = {
    "BENIGN": 0,
    "DDOS FLOOD": 0,
    "DOS HULK": 0,
    "BOTNET COMMAND": 0,
    "INFILTRATION PROBE": 0,
    "SYN PORTSCAN": 0,
    "SSH BRUTE FORCE": 0,
    "WEB SQL INJECTION": 0
};

// 3. Real-Time Packet Sniffer Simulation
function setupSniffer() {
    const startBtn = document.getElementById("start-btn");
    const stopBtn = document.getElementById("stop-btn");
    const clearBtn = document.getElementById("clear-log");

    startBtn.addEventListener("click", () => {
        isSniffing = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        logTerminal("SYSTEM CONTROLS: Commencing network threat packet monitoring...", "blue");
        
        // Start simulated packet generation loop
        snifferInterval = setInterval(() => {
            generateSimulatedPacket();
        }, 1200);
    });

    stopBtn.addEventListener("click", () => {
        isSniffing = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        clearInterval(snifferInterval);
        logTerminal("SYSTEM CONTROLS: Network monitoring stopped by operator.", "yellow");
    });

    clearBtn.addEventListener("click", () => {
        threatLog = [];
        document.getElementById("live-log-body").innerHTML = `
            <tr class="empty-row">
                <td colspan="7">Ready to scan. Please click 'Start Threat Monitoring' in the control panel.</td>
            </tr>
        `;
        logTerminal("SOC LOGS: Active threat log registry cleared.", "muted");
    });
}

function generateSimulatedPacket() {
    // 90% benign traffic, 10% chance of threat injection
    const rand = Math.random();
    let typeKey = "benign";
    
    if (rand > 0.90) {
        const threats = ["ddos", "dos", "bot", "infiltration", "portscan", "bruteforce", "webattack"];
        typeKey = threats[Math.floor(Math.random() * threats.length)];
    }

    injectPacketData(typeKey);
}

function injectPacketData(typeKey) {
    const attack = ATTACK_TYPES[typeKey];
    const score = randomFloat(attack.scoreRange[0], attack.scoreRange[1]);
    const confidence = randomFloat(attack.confidenceRange[0], attack.confidenceRange[1]);
    const threshold = 0.05284;
    const isAnom = score > threshold;

    totalPackets++;
    anomalySum += score;
    
    if (attack.name !== "BENIGN" || isAnom) {
        threatsCount++;
        threatCounts[attack.name] = (threatCounts[attack.name] || 0) + 1;
    } else {
        threatCounts["BENIGN"]++;
    }

    // Append to live logs array
    const timestamp = new Date().toISOString().replace("T", " ").slice(0, 23);
    const logEntry = {
        timestamp,
        score: score.toFixed(5),
        isAnomaly: isAnom,
        classification: attack.name,
        confidence: confidence.toFixed(2) + "%",
        severity: attack.severity,
        mitigation: attack.name === "BENIGN" ? "Ignore" : attack.action
    };
    
    threatLog.unshift(logEntry);
    if (threatLog.length > 50) threatLog.pop();

    // Update UI Stats
    document.getElementById("metric-total").innerText = totalPackets.toLocaleString();
    document.getElementById("metric-threats").innerText = threatsCount.toLocaleString();
    document.getElementById("metric-anomaly").innerText = (anomalySum / totalPackets).toFixed(4);
    
    const threatRatio = ((threatsCount / totalPackets) * 100).toFixed(2);
    document.getElementById("metric-threat-ratio").innerText = `${threatRatio}% anomaly rate`;

    const rate = isSniffing ? Math.floor(Math.random() * 15) + 35 : 0;
    document.getElementById("metric-rate").innerText = `${rate} pkts/sec`;

    // Render Table
    renderTable();

    // Update Charts
    updateChartsRealtime(score);

    // Trigger log alerts
    if (attack.name !== "BENIGN") {
        logTerminal(`[WARNING] Intrusion Detected: ${attack.name} (Conf: ${logEntry.confidence}, Action: ${logEntry.mitigation})`, "red");
    } else {
        if (totalPackets % 5 === 0) {
            logTerminal(`[INFO] Flow processed successfully: BENIGN stream packet verified.`, "green");
        }
    }
    
    // Update system load simulation values
    updateSystemResources();
}

function renderTable() {
    const tbody = document.getElementById("live-log-body");
    if (threatLog.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-row">
                <td colspan="7">Ready to scan. Please click 'Start Threat Monitoring' in the control panel.</td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = threatLog.map(entry => `
        <tr>
            <td class="font-mono text-secondary">${entry.timestamp}</td>
            <td class="font-mono">${entry.score}</td>
            <td>
                <span class="badge ${entry.isAnomaly ? 'badge-anomaly' : 'badge-normal'}">
                    ${entry.isAnomaly ? 'ANOMALOUS' : 'NORMAL'}
                </span>
            </td>
            <td class="font-mono ${entry.classification !== 'BENIGN' ? 'text-red' : ''}">${entry.classification}</td>
            <td class="font-mono">${entry.confidence}</td>
            <td class="severity-${entry.severity.toLowerCase()}">${entry.severity}</td>
            <td class="font-mono text-green">${entry.mitigation}</td>
        </tr>
    `).join("");
}

function updateSystemResources() {
    const cpu = isSniffing ? Math.floor(Math.random() * 15) + 12 : 5;
    const vram = isSniffing ? (3.8 + Math.random() * 0.8).toFixed(1) : "1.2";
    const latency = isSniffing ? (0.65 + Math.random() * 0.4).toFixed(2) : "0.00";

    document.getElementById("cpu-pct").innerText = `${cpu}%`;
    document.getElementById("cpu-bar").style.width = `${cpu}%`;
    
    document.getElementById("vram-pct").innerText = `${vram} GB`;
    document.getElementById("vram-bar").style.width = `${(parseFloat(vram) / 12) * 100}%`;

    document.getElementById("latency-ms").innerText = `${latency} ms`;
    document.getElementById("latency-bar").style.width = `${(parseFloat(latency) / 5) * 100}%`;
}

// 4. Threat Simulator Module
function setupSimulator() {
    const simButtons = document.querySelectorAll(".btn-sim");
    
    simButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const attackType = btn.getAttribute("data-attack");
            runSimulatorPipeline(attackType);
        });
    });

    const injectCustomBtn = document.getElementById("inject-custom-btn");
    injectCustomBtn.addEventListener("click", () => {
        // Collect custom fields
        const port = parseInt(document.getElementById("custom-port").value) || 80;
        const rate = parseInt(document.getElementById("custom-rate").value) || 100;
        
        let attackKey = "benign";
        if (rate > 5000 || port === 6667) {
            attackKey = "ddos";
        } else if (port === 22 || port === 23) {
            attackKey = "bruteforce";
        } else if (port > 1000 && Math.random() > 0.5) {
            attackKey = "portscan";
        }

        runSimulatorPipeline(attackKey);
    });
}

function runSimulatorPipeline(attackKey) {
    const attack = ATTACK_TYPES[attackKey];
    const score = randomFloat(attack.scoreRange[0], attack.scoreRange[1]);
    const confidence = randomFloat(attack.confidenceRange[0], attack.confidenceRange[1]);
    const threshold = 0.05284;
    const isAnom = score > threshold;

    // Show Result card
    document.getElementById("sim-result-placeholder").classList.add("hidden");
    const details = document.getElementById("sim-result-details");
    details.classList.remove("hidden");

    // Configure Verdict Banner style
    const verdict = document.getElementById("sim-verdict");
    const classVal = document.getElementById("sim-attack-class");
    const descVal = document.getElementById("sim-attack-desc");

    classVal.innerText = attack.name;
    descVal.innerText = attack.desc;

    if (attack.name === "BENIGN") {
        verdict.className = "sim-verdict-card benign";
        classVal.className = "verdict-value text-green";
    } else {
        verdict.className = "sim-verdict-card";
        classVal.className = "verdict-value text-red";
    }

    // Configure numeric stats
    document.getElementById("sim-anomaly-score").innerText = score.toFixed(5);
    document.getElementById("sim-confidence").innerText = confidence.toFixed(2) + "%";
    document.getElementById("sim-mitigation").innerText = attack.name === "BENIGN" ? "Ignore" : attack.action;

    // Set mock local SHAP contributions
    renderSimShapBars(attackKey);

    // Feed back into the live logs automatically
    injectPacketData(attackKey);
}

function renderSimShapBars(attackKey) {
    const shapContainer = document.getElementById("sim-shap-bars");
    
    // Dynamic feature coefficients matching the attack profile
    let values = {};
    if (attackKey === "ddos" || attackKey === "dos") {
        values = { "flow_packets_s": 0.42, "flow_duration": 0.18, "fwd_pkt_len_max": 0.12, "destination_port": -0.05 };
    } else if (attackKey === "portscan") {
        values = { "destination_port": 0.48, "flow_iat_mean": -0.22, "flow_packets_s": 0.10, "flow_duration": -0.08 };
    } else if (attackKey === "bot") {
        values = { "flow_iat_mean": -0.38, "destination_port": 0.15, "active_mean": 0.08, "idle_mean": -0.05 };
    } else if (attackKey === "bruteforce") {
        values = { "destination_port": 0.32, "flow_packets_s": 0.22, "flow_duration": 0.15, "fwd_pkt_len_max": -0.05 };
    } else {
        // Benign/Default
        values = { "destination_port": -0.12, "flow_packets_s": -0.08, "flow_duration": -0.05, "flow_iat_mean": 0.04 };
    }

    shapContainer.innerHTML = Object.entries(values).map(([feature, val]) => {
        const isPos = val >= 0;
        const absPct = Math.min(Math.abs(val) * 150, 100); // Scaled for chart representation
        return `
            <div class="shap-bar-row">
                <div class="bar-lbl">${feature}</div>
                <div class="bar-track">
                    <div class="bar-fill ${isPos ? 'positive' : 'negative'}" style="width: ${absPct}%"></div>
                </div>
                <div class="bar-val ${isPos ? 'text-red' : 'text-blue'}">${isPos ? '+' : ''}${val.toFixed(2)}</div>
            </div>
        `;
    }).join("");
}

// 5. Model Governance & Feature Explainability Panels
function setupXaiViewer() {
    const wrapper = document.getElementById("xai-features-wrapper");
    
    // Add feature badges
    wrapper.innerHTML = Object.keys(FEATURE_DATA).map((feat, idx) => `
        <span class="xai-feature-badge ${idx === 0 ? 'active' : ''}" data-feature="${feat}">
            ${feat}
        </span>
    `).join("");

    const badges = document.querySelectorAll(".xai-feature-badge");
    badges.forEach(badge => {
        badge.addEventListener("click", () => {
            badges.forEach(b => b.classList.remove("active"));
            badge.classList.add("active");
            
            const featureName = badge.getAttribute("data-feature");
            updateAttributionCard(featureName);
        });
    });

    // Initialize with first feature
    updateAttributionCard(Object.keys(FEATURE_DATA)[0]);
}

function updateAttributionCard(name) {
    const data = FEATURE_DATA[name];
    document.getElementById("attrib-feature-name").innerText = name.toUpperCase();
    document.getElementById("attrib-feature-desc").innerText = data.desc;
    
    const capVal = document.getElementById("attrib-captum");
    capVal.innerText = (data.captum >= 0 ? '+' : '') + data.captum.toFixed(4);
    capVal.className = `box-val font-mono ${data.captum >= 0 ? 'text-red' : 'text-blue'}`;

    const shapVal = document.getElementById("attrib-shap");
    shapVal.innerText = (data.shap >= 0 ? '+' : '') + data.shap.toFixed(4);
    shapVal.className = `box-val font-mono ${data.shap >= 0 ? 'text-red' : 'text-blue'}`;
}

// 6. Chart.js Dashboard Visualizations
function setupCharts() {
    // Live Anomaly line chart
    const ctxLine = document.getElementById("anomalyChart").getContext("2d");
    anomalyChart = new Chart(ctxLine, {
        type: 'line',
        data: {
            labels: anomalyLabels,
            datasets: [
                {
                    label: 'Anomaly Score (Reconstruction MSE)',
                    data: anomalyDataPoints,
                    borderColor: '#ff0055',
                    backgroundColor: 'rgba(255, 0, 85, 0.05)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 1
                },
                {
                    label: 'Anomaly Threshold (0.05284)',
                    data: Array(20).fill(0.05284),
                    borderColor: '#ffcc00',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#8b949e', font: { family: 'JetBrains Mono', size: 9 } }
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 0.4,
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#8b949e', font: { family: 'JetBrains Mono', size: 9 } }
                },
                x: {
                    grid: { display: false },
                    ticks: { display: false }
                }
            }
        }
    });

    // Threat Distribution pie/doughnut chart
    const ctxDoughnut = document.getElementById("threatDistributionChart").getContext("2d");
    threatDistChart = new Chart(ctxDoughnut, {
        type: 'doughnut',
        data: {
            labels: ["Benign", "DDoS", "DoS Hulk", "PortScan", "Brute Force", "Others"],
            datasets: [{
                data: [1, 0, 0, 0, 0, 0],
                backgroundColor: [
                    '#00ff66',
                    '#ff0055',
                    '#ff5500',
                    '#00f0ff',
                    '#bd00ff',
                    '#ffcc00'
                ],
                borderWidth: 1,
                borderColor: '#0d1321'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#8b949e', font: { size: 9 } }
                }
            }
        }
    });

    // Global Feature Importance Bar Chart
    const ctxGlobal = document.getElementById("globalImportanceChart").getContext("2d");
    globalImportanceChart = new Chart(ctxGlobal, {
        type: 'bar',
        data: {
            labels: [
                "Destination Port", 
                "Flow Packets/s", 
                "Flow IAT Mean", 
                "Fwd Pkt Len Max", 
                "Bwd Pkt Len Min", 
                "Active Mean", 
                "Flow Duration", 
                "Idle Mean"
            ],
            datasets: [{
                label: 'XGBoost Feature Importance Weight',
                data: [0.245, 0.210, 0.165, 0.142, 0.098, 0.065, 0.045, 0.030],
                backgroundColor: 'rgba(0, 240, 255, 0.25)',
                borderColor: '#00f0ff',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#8b949e', font: { family: 'JetBrains Mono', size: 9 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f0f6fc', font: { size: 10 } }
                }
            }
        }
    });
}

function updateChartsRealtime(newScore) {
    // 1. Line Chart Scoring
    anomalyDataPoints.push(newScore);
    anomalyDataPoints.shift();
    anomalyChart.update('none');

    // 2. Doughnut Distribution Chart
    const others = threatCounts["BOTNET COMMAND"] + threatCounts["INFILTRATION PROBE"] + threatCounts["WEB SQL INJECTION"];
    threatDistChart.data.datasets[0].data = [
        threatCounts["BENIGN"],
        threatCounts["DDOS FLOOD"],
        threatCounts["DOS HULK"],
        threatCounts["SYN PORTSCAN"],
        threatCounts["SSH BRUTE FORCE"],
        others
    ];
    threatDistChart.update('none');
}

// 7. Audit Exporters & Reporting
function setupReporting() {
    const csvBtn = document.getElementById("export-csv-btn");
    const pdfBtn = document.getElementById("gen-pdf-btn");

    csvBtn.addEventListener("click", () => {
        if (threatLog.length === 0) {
            alert("No threat records to export. Please start the monitoring sniffer feed first.");
            return;
        }

        // Compile CSV headers & records
        const headers = "Timestamp,Anomaly Score,Classification,Confidence,Severity,RL Mitigation\n";
        const rows = threatLog.map(r => 
            `"${r.timestamp}","${r.score}","${r.classification}","${r.confidence}","${r.severity}","${r.mitigation}"`
        ).join("\n");

        const blob = new Blob([headers + rows], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", url);
        link.setAttribute("download", `threat_audit_report_${new Date().toISOString().slice(0,10)}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        logTerminal("SOC REPORT: Exported secure CSV audit registry.", "green");
    });

    pdfBtn.addEventListener("click", () => {
        window.print();
    });
}

// Helper utilities
function randomFloat(min, max) {
    return Math.random() * (max - min) + min;
}

function logTerminal(text, type = "muted") {
    const terminal = document.getElementById("audit-log-terminal");
    const line = document.createElement("div");
    const now = new Date().toISOString().replace("T", " ").slice(0, 19);
    line.className = `terminal-line ${type}`;
    line.innerText = `[${now}] ${text}`;
    terminal.appendChild(line);
    terminal.scrollTop = terminal.scrollHeight;
}
