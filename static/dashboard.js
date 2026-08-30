/**
 * DigitalTwin.ai — Dashboard Frontend
 * =====================================
 * Drives the interactive multi-stakeholder dashboard.
 * - Fetches data from Flask API
 * - Renders assembly graph with D3.js-style SVG
 * - Updates KPIs, alerts, heatmaps, and charts
 * - Handles what-if simulation
 * - Tab navigation and auto-advance
 *
 * Author : Piyush Maji
 * Project: Accenture Innovation Challenge 2026
 */

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBALS
// ═══════════════════════════════════════════════════════════════════════════════
let config = null;         // Static config (stations, edges, etc.)
let currentState = null;   // Current twin state
let timelineData = [];     // Accumulated timeline
let autoPlayInterval = null;
let isAutoPlaying = false;

const RISK_COLORS = ['#16a34a', '#d97706', '#dc2626'];
const RISK_LABELS = ['LOW', 'MEDIUM', 'HIGH'];
const ZONE_COLORS = {
    'Body Construction': '#0284c7',
    'Paint Shop': '#7c3aed',
    'Final Assembly': '#059669'
};

// ═══════════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {
    await loadConfig();
    await loadState();
    setupTabs();
    setupControls();
    renderAll();
});

async function loadConfig() {
    const res = await fetch('/api/config');
    config = await res.json();
}

async function loadState() {
    const res = await fetch('/api/state');
    currentState = await res.json();
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════════

function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
        });
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMULATION CONTROLS
// ═══════════════════════════════════════════════════════════════════════════════

function setupControls() {
    document.getElementById('btn-step').addEventListener('click', advanceStep);
    document.getElementById('btn-autoplay').addEventListener('click', toggleAutoPlay);
    document.getElementById('btn-reset').addEventListener('click', resetSimulation);
    document.getElementById('btn-run-whatif').addEventListener('click', runWhatIf);
}

async function advanceStep() {
    const res = await fetch('/api/step', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'ok') {
        currentState = { step: data.step, predictions: data.predictions };
        timelineData.push(data.predictions);
        renderAll();
        updateProgress(data.step, config.stations.length > 0 ? 29 : 29);
    } else if (data.status === 'end_of_shift') {
        stopAutoPlay();
        showNotification('End of shift reached', 'info');
    }
}

function toggleAutoPlay() {
    if (isAutoPlaying) {
        stopAutoPlay();
    } else {
        isAutoPlaying = true;
        document.getElementById('btn-autoplay').textContent = '⏸ Pause';
        document.getElementById('btn-autoplay').classList.add('primary');
        autoPlayInterval = setInterval(advanceStep, 1200);
    }
}

function stopAutoPlay() {
    isAutoPlaying = false;
    clearInterval(autoPlayInterval);
    document.getElementById('btn-autoplay').textContent = '▶ Auto-Play';
    document.getElementById('btn-autoplay').classList.remove('primary');
}

async function resetSimulation() {
    stopAutoPlay();
    await fetch('/api/reset', { method: 'POST' });
    timelineData = [];
    await loadState();
    renderAll();
    updateProgress(0, 29);
    showNotification('Simulation reset', 'info');
}

function updateProgress(step, total) {
    const pct = (step / total) * 100;
    document.getElementById('progress-fill').style.width = `${pct}%`;
    document.getElementById('progress-text').textContent = `Step ${step}/${total} — T+${step * 16}min`;
}

function showNotification(msg, type) {
    const el = document.getElementById('notification');
    el.textContent = msg;
    el.className = `status-badge ${type === 'info' ? 'live' : ''}`;
    el.style.display = 'flex';
    setTimeout(() => { el.style.display = 'none'; }, 3000);
}

// ═══════════════════════════════════════════════════════════════════════════════
// RENDER ALL VIEWS
// ═══════════════════════════════════════════════════════════════════════════════

function renderAll() {
    if (!config || !currentState) return;
    renderFloorView();
    renderPlannerView();
    renderLeadershipView();
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB 1: FLOOR SUPERVISOR VIEW
// ═══════════════════════════════════════════════════════════════════════════════

function renderFloorView() {
    renderAssemblyGraph();
    renderAlerts();
    renderStationTable();
    renderVirtualSensing();
}

function renderAssemblyGraph() {
    const container = document.getElementById('graph-svg');
    if (!container) return;

    const pred = currentState.predictions;
    const width = container.clientWidth || 900;
    const height = 420;

    // Compute layout bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    config.stations.forEach(s => {
        const [x, y] = s.pos;
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    });

    const padX = 60, padY = 50;
    const scaleX = (width - 2 * padX) / (maxX - minX || 1);
    const scaleY = (height - 2 * padY) / (maxY - minY || 1);

    const px = (x) => padX + (x - minX) * scaleX;
    const py = (y) => padY + (maxY - y) * scaleY;  // flip Y

    let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;

    // Defs for arrow markers
    svg += `<defs>
        <marker id="arrow-fwd" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#0284c7" opacity="0.7"/>
        </marker>
        <marker id="arrow-fb" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#db2777" opacity="0.5"/>
        </marker>
    </defs>`;

    // Draw edges
    config.edges.forEach(e => {
        const s1 = config.stations[e.source];
        const s2 = config.stations[e.target];
        const x1 = px(s1.pos[0]), y1 = py(s1.pos[1]);
        const x2 = px(s2.pos[0]), y2 = py(s2.pos[1]);

        if (e.type === 'forward') {
            svg += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" 
                     stroke="#0284c7" stroke-width="1.5" opacity="0.35" marker-end="url(#arrow-fwd)"/>`;
        } else {
            // Curved feedback edge
            const mx = (x1 + x2) / 2;
            const my = (y1 + y2) / 2 - 30;
            svg += `<path d="M${x1},${y1} Q${mx},${my} ${x2},${y2}" 
                     fill="none" stroke="#db2777" stroke-width="1" stroke-dasharray="4,3" 
                     opacity="0.3" marker-end="url(#arrow-fb)"/>`;
        }
    });

    // Draw nodes
    config.stations.forEach((s, i) => {
        const cx = px(s.pos[0]);
        const cy = py(s.pos[1]);
        const risk = pred.risk_class[i];
        const queue = pred.queue[i];
        const anomaly = pred.anomaly_score[i];
        const color = RISK_COLORS[risk];
        const radius = Math.max(10, Math.min(22, 8 + queue * 0.8));

        // Glow for high-risk
        if (risk === 2) {
            svg += `<circle cx="${cx}" cy="${cy}" r="${radius + 6}" fill="none" 
                     stroke="${color}" stroke-width="2" opacity="0.3">
                     <animate attributeName="r" values="${radius+4};${radius+10};${radius+4}" dur="2s" repeatCount="indefinite"/>
                     <animate attributeName="opacity" values="0.3;0.1;0.3" dur="2s" repeatCount="indefinite"/>
                    </circle>`;
        }

        // Node circle
        svg += `<circle cx="${cx}" cy="${cy}" r="${radius}" fill="${color}" opacity="0.85" 
                 stroke="#fff" stroke-width="1.5"/>`;

        // Label
        svg += `<text x="${cx}" y="${cy + radius + 14}" text-anchor="middle" 
                 fill="#475569" font-size="8" font-family="Inter, sans-serif">${s.short}</text>`;

        // Queue value on node
        svg += `<text x="${cx}" y="${cy + 3}" text-anchor="middle" fill="white" 
                 font-size="7" font-weight="bold" font-family="monospace">Q:${Math.round(queue)}</text>`;
    });

    // Zone labels
    const zones = [
        { label: 'BODY CONSTRUCTION', color: '#0284c7', y: 12 },
        { label: 'PAINT SHOP', color: '#7c3aed', y: height / 2 - 10 },
        { label: 'FINAL ASSEMBLY', color: '#059669', y: height - 25 },
    ];
    zones.forEach(z => {
        svg += `<text x="8" y="${z.y}" fill="${z.color}" font-size="9" font-weight="700" 
                 opacity="0.6" font-family="Inter, sans-serif">${z.label}</text>`;
    });

    svg += '</svg>';
    container.innerHTML = svg;
}

function renderAlerts() {
    const container = document.getElementById('alerts-container');
    if (!container || !currentState.predictions) return;

    const pred = currentState.predictions;
    const alerts = [];

    config.stations.forEach((s, i) => {
        if (pred.risk_class[i] >= 1) {
            alerts.push({
                station: s,
                risk: pred.risk_class[i],
                queue: pred.queue[i],
                anomaly: pred.anomaly_score[i],
                confidence: pred.confidence[i],
            });
        }
    });

    // Sort by risk (high first), then by anomaly score
    alerts.sort((a, b) => b.risk - a.risk || b.anomaly - a.anomaly);

    if (alerts.length === 0) {
        container.innerHTML = `<div class="empty-state">
            <div class="empty-state-icon">✅</div>
            <p>All stations operating normally</p>
        </div>`;
        return;
    }

    container.innerHTML = alerts.slice(0, 6).map(a => {
        const level = a.risk === 2 ? 'high' : 'medium';
        const icon = a.risk === 2 ? '🔴' : '🟡';
        const riskLabel = RISK_LABELS[a.risk];
        return `<div class="alert-card ${level}">
            <span class="alert-icon">${icon}</span>
            <div>
                <div class="alert-title">${a.station.name}</div>
                <div class="alert-detail">
                    Risk: <strong>${riskLabel}</strong> · 
                    Queue: ${a.queue.toFixed(0)} · 
                    Anomaly: ${(a.anomaly * 100).toFixed(0)}% · 
                    Zone: ${a.station.zone}
                </div>
                <div class="alert-detail text-muted" style="margin-top:2px">
                    ${a.station.instrumentation.toUpperCase()} instrumentation · 
                    ${a.station.vintage} equipment
                </div>
            </div>
        </div>`;
    }).join('');
}

function renderStationTable() {
    const tbody = document.getElementById('station-tbody');
    if (!tbody) return;

    const pred = currentState.predictions;

    tbody.innerHTML = config.stations.map((s, i) => {
        const risk = pred.risk_class[i];
        const riskLabel = RISK_LABELS[risk].toLowerCase();
        const conf = pred.confidence[i];
        const confClass = conf >= 0.75 ? 'high' : conf >= 0.5 ? 'medium' : 'low';
        const zoneClass = s.zone.includes('Body') ? 'body' : s.zone.includes('Paint') ? 'paint' : 'final';

        return `<tr>
            <td><span class="font-mono">${String(s.id).padStart(2, '0')}</span></td>
            <td><strong>${s.name}</strong></td>
            <td><span class="zone-badge ${zoneClass}">${s.zone.split(' ')[0]}</span></td>
            <td><span class="instr-badge ${s.instrumentation}">${s.instrumentation}</span></td>
            <td><span class="risk-badge ${riskLabel}">${RISK_LABELS[risk]}</span></td>
            <td class="font-mono">${pred.queue[i].toFixed(1)}</td>
            <td class="font-mono">${(pred.anomaly_score[i] * 100).toFixed(0)}%</td>
            <td>
                <div class="confidence-bar">
                    <div class="confidence-track">
                        <div class="confidence-fill ${confClass}" style="width:${conf * 100}%"></div>
                    </div>
                    <span class="confidence-value">${(conf * 100).toFixed(0)}%</span>
                </div>
            </td>
        </tr>`;
    }).join('');
}

function renderVirtualSensing() {
    const container = document.getElementById('virtual-sensing');
    if (!container) return;

    const pred = currentState.predictions;

    // Find stations with low confidence (under-instrumented)
    const lowConf = config.stations
        .map((s, i) => ({ station: s, confidence: pred.confidence[i], idx: i }))
        .filter(d => d.station.instrumentation !== 'full')
        .sort((a, b) => a.confidence - b.confidence);

    container.innerHTML = lowConf.map(d => {
        const confPct = (d.confidence * 100).toFixed(0);
        const confClass = d.confidence >= 0.75 ? 'high' : d.confidence >= 0.5 ? 'medium' : 'low';
        const needsInspection = d.confidence < 0.5;

        return `<div class="scenario-card ${needsInspection ? 'active' : ''}">
            <div class="flex justify-between items-center">
                <span class="scenario-name">${d.station.name}</span>
                <span class="instr-badge ${d.station.instrumentation}">${d.station.instrumentation}</span>
            </div>
            <div class="confidence-bar mt-md">
                <div class="confidence-track" style="height:8px">
                    <div class="confidence-fill ${confClass}" style="width:${confPct}%"></div>
                </div>
                <span class="confidence-value ${confClass === 'low' ? 'text-danger' : ''}">${confPct}%</span>
            </div>
            ${needsInspection ? '<div class="alert-detail text-danger" style="margin-top:4px">⚠ Flag for manual inspection</div>' : ''}
        </div>`;
    }).join('');
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB 2: PLANT MANAGER VIEW
// ═══════════════════════════════════════════════════════════════════════════════

function renderPlannerView() {
    renderShiftKPIs();
    renderRiskHeatmap();
    renderBottleneckChart();
    renderScenarioStatus();
}

function renderShiftKPIs() {
    if (!currentState.predictions) return;
    const pred = currentState.predictions;

    const highRisk = pred.risk_class.filter(r => r === 2).length;
    const medRisk = pred.risk_class.filter(r => r === 1).length;
    const avgQueue = pred.queue.reduce((a, b) => a + b, 0) / pred.queue.length;
    const avgAnomaly = pred.anomaly_score.reduce((a, b) => a + b, 0) / pred.anomaly_score.length;
    const lowConfCount = pred.confidence.filter(c => c < 0.5).length;

    // Estimate OEE (simplified)
    const avgUtil = pred.features
        ? pred.features.reduce((sum, f) => sum + f[6], 0) / pred.features.length
        : 75;
    const defectFree = 1 - (pred.risk_class.filter(r => r >= 1).length / config.stations.length);
    const oee = (avgUtil / 100) * defectFree * 0.92;  // availability * quality * performance

    setKPI('kpi-high-risk', highRisk, highRisk > 3 ? 'negative' : 'positive');
    setKPI('kpi-med-risk', medRisk);
    setKPI('kpi-avg-queue', avgQueue.toFixed(1));
    setKPI('kpi-oee', (oee * 100).toFixed(1) + '%', oee > 0.7 ? 'positive' : 'negative');
    setKPI('kpi-low-conf', lowConfCount, lowConfCount > 5 ? 'negative' : 'positive');
    setKPI('kpi-anomaly', (avgAnomaly * 100).toFixed(0) + '%');
}

function setKPI(id, value, deltaClass) {
    const el = document.getElementById(id);
    if (el) {
        el.querySelector('.kpi-value').textContent = value;
        if (deltaClass) {
            const deltaEl = el.querySelector('.kpi-delta');
            if (deltaEl) deltaEl.className = `kpi-delta ${deltaClass}`;
        }
    }
}

function renderRiskHeatmap() {
    const container = document.getElementById('risk-heatmap');
    if (!container || timelineData.length < 1) {
        if (container) container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📊</div><p>Advance simulation to see risk heatmap</p></div>';
        return;
    }

    const steps = timelineData.length;
    const numStations = config.stations.length;

    let html = '<div class="heatmap-grid" style="grid-template-columns: 100px repeat(' + steps + ', 1fr)">';

    // Header row
    html += '<div style="font-size:0.65rem;color:var(--text-muted)">Station</div>';
    for (let t = 0; t < steps; t++) {
        html += `<div style="font-size:0.55rem;color:var(--text-muted);text-align:center">T${t + 1}</div>`;
    }

    // Data rows — show top 15 most interesting stations
    const stationInterest = config.stations.map((s, i) => {
        const maxRisk = Math.max(...timelineData.map(d => d.risk_class[i]));
        const maxAnomaly = Math.max(...timelineData.map(d => d.anomaly_score[i]));
        return { idx: i, interest: maxRisk * 3 + maxAnomaly };
    }).sort((a, b) => b.interest - a.interest).slice(0, 15);

    stationInterest.forEach(({ idx }) => {
        const s = config.stations[idx];
        html += `<div style="font-size:0.65rem;color:var(--text-secondary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:2px 4px">${s.short}</div>`;
        for (let t = 0; t < steps; t++) {
            const risk = timelineData[t].risk_class[idx];
            const color = RISK_COLORS[risk];
            const opacity = 0.4 + (timelineData[t].anomaly_score[idx]) * 0.6;
            html += `<div class="heatmap-cell" style="background:${color};opacity:${opacity}" 
                      title="${s.name} @ T${t + 1}: ${RISK_LABELS[risk]}"></div>`;
        }
    });

    html += '</div>';
    container.innerHTML = html;
}

function renderBottleneckChart() {
    const container = document.getElementById('bottleneck-chart');
    if (!container || timelineData.length < 2) return;

    // Count how often each station was medium or high risk
    const riskCounts = new Array(config.stations.length).fill(0);
    timelineData.forEach(d => {
        d.risk_class.forEach((r, i) => { if (r >= 1) riskCounts[i]++; });
    });

    const ranked = riskCounts
        .map((count, i) => ({ station: config.stations[i], count, idx: i }))
        .filter(d => d.count > 0)
        .sort((a, b) => b.count - a.count)
        .slice(0, 10);

    if (ranked.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>No bottlenecks detected yet</p></div>';
        return;
    }

    const maxCount = ranked[0].count;
    container.innerHTML = ranked.map(d => {
        const pct = (d.count / maxCount) * 100;
        const zoneClass = d.station.zone.includes('Body') ? 'body' : d.station.zone.includes('Paint') ? 'paint' : 'final';
        return `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
            <span style="min-width:100px;font-size:0.75rem;color:var(--text-secondary)">${d.station.short}</span>
            <div style="flex:1;height:16px;background:var(--bg-glass);border-radius:4px;overflow:hidden">
                <div style="height:100%;width:${pct}%;background:${ZONE_COLORS[d.station.zone]};opacity:0.7;border-radius:4px;transition:width 0.5s"></div>
            </div>
            <span class="font-mono" style="min-width:30px;font-size:0.75rem;color:var(--text-muted)">${d.count}x</span>
        </div>`;
    }).join('');
}

function renderScenarioStatus() {
    const container = document.getElementById('scenario-status');
    if (!container) return;

    const activeScenarios = currentState.predictions?.active_scenarios || [];

    container.innerHTML = config.scenarios.map(s => {
        const isActive = activeScenarios.includes(s.name);
        return `<div class="scenario-card ${isActive ? 'active' : ''}">
            <div class="flex justify-between items-center">
                <span class="scenario-name">${isActive ? '🔴' : '⚪'} ${s.name}</span>
                <span class="card-badge" style="background:${isActive ? 'var(--risk-high-bg)' : 'var(--bg-glass)'};color:${isActive ? 'var(--risk-high)' : 'var(--text-muted)'}">
                    ${isActive ? 'ACTIVE' : 'INACTIVE'}
                </span>
            </div>
            <div class="scenario-meta mt-md">${s.description}</div>
            <div class="scenario-meta">Category: ${s.category.replace('_', ' ')} · Onset: Step ${s.onset}</div>
        </div>`;
    }).join('');
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAB 3: LEADERSHIP VIEW
// ═══════════════════════════════════════════════════════════════════════════════

function renderLeadershipView() {
    renderROICards();
    renderValidationMetrics();
    renderMultiPlantView();
}

function renderROICards() {
    // These are projected values based on model capabilities
    const projections = [
        { label: 'Annual Downtime Reduction', value: '35%', delta: '↓ $1.2M/yr', deltaClass: 'positive' },
        { label: 'Rework Cost Savings', value: '$2.8M', delta: 'per plant/year', deltaClass: 'positive' },
        { label: 'Bottleneck Lead Time', value: '12 min', delta: 'advance warning', deltaClass: 'positive' },
        { label: 'Sensor HW Savings', value: '45%', delta: 'via virtual sensing', deltaClass: 'positive' },
    ];

    const container = document.getElementById('roi-cards');
    if (!container) return;

    container.innerHTML = projections.map(p => `
        <div class="kpi-card">
            <div class="kpi-value text-accent">${p.value}</div>
            <div class="kpi-label">${p.label}</div>
            <div class="kpi-delta ${p.deltaClass}">${p.delta}</div>
        </div>
    `).join('');
}

async function renderValidationMetrics() {
    const container = document.getElementById('validation-metrics');
    if (!container) return;

    try {
        const res = await fetch('/api/validation');
        const data = await res.json();

        if (data.status === 'insufficient_data') {
            container.innerHTML = `<div class="empty-state"><div class="empty-state-icon">📈</div>
                <p>Advance simulation to at least 3 steps for validation metrics</p></div>`;
            return;
        }

        const m = data.metrics;
        container.innerHTML = `
            <div class="grid-3 mb-lg">
                <div class="kpi-card">
                    <div class="kpi-value ${m.precision > 0.7 ? 'text-success' : 'text-warning'}">${(m.precision * 100).toFixed(1)}%</div>
                    <div class="kpi-label">Precision</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value ${m.recall > 0.7 ? 'text-success' : 'text-warning'}">${(m.recall * 100).toFixed(1)}%</div>
                    <div class="kpi-label">Recall</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value ${m.false_alarm_rate < 0.1 ? 'text-success' : 'text-danger'}">${(m.false_alarm_rate * 100).toFixed(1)}%</div>
                    <div class="kpi-label">False Alarm Rate</div>
                </div>
            </div>
            <div class="grid-2">
                <div>
                    <h4 style="font-size:0.85rem;margin-bottom:8px">Confusion Matrix</h4>
                    <table class="station-table" style="font-size:0.75rem">
                        <tr><th></th><th>Pred Anomaly</th><th>Pred Normal</th></tr>
                        <tr><td><strong>Actual Anomaly</strong></td>
                            <td class="text-success font-mono">${data.confusion.tp} TP</td>
                            <td class="text-danger font-mono">${data.confusion.fn} FN</td></tr>
                        <tr><td><strong>Actual Normal</strong></td>
                            <td class="text-warning font-mono">${data.confusion.fp} FP</td>
                            <td class="text-success font-mono">${data.confusion.tn} TN</td></tr>
                    </table>
                </div>
                <div>
                    <h4 style="font-size:0.85rem;margin-bottom:8px">Evaluation Summary</h4>
                    <div class="alert-detail">
                        <p>• ${data.steps_evaluated} time steps evaluated</p>
                        <p>• ${data.total_predictions} station-predictions assessed</p>
                        <p>• F1 Score: <strong class="text-accent">${(m.f1_score * 100).toFixed(1)}%</strong></p>
                        <p>• Accuracy: <strong>${(m.accuracy * 100).toFixed(1)}%</strong></p>
                    </div>
                </div>
            </div>
        `;
    } catch (e) {
        container.innerHTML = '<div class="empty-state"><p>Error loading validation data</p></div>';
    }
}

function renderMultiPlantView() {
    const container = document.getElementById('multi-plant');
    if (!container) return;

    const plants = [
        { name: 'Plant A — Pilot', location: 'Chennai, India', stations: 40, maturity: 'High', coverage: '60%', savings: '$2.8M', status: 'Active' },
        { name: 'Plant B — Rollout', location: 'Pune, India', stations: 35, maturity: 'Medium', coverage: '45%', savings: '$2.1M', status: 'Planned Q2' },
        { name: 'Plant C — Future', location: 'Detroit, USA', stations: 52, maturity: 'Low', coverage: '30%', savings: '$3.4M', status: 'Planned Q4' },
    ];

    container.innerHTML = `<table class="station-table">
        <thead><tr>
            <th>Plant</th><th>Location</th><th>Stations</th><th>Sensor Maturity</th>
            <th>Full Coverage</th><th>Est. Annual Savings</th><th>Status</th>
        </tr></thead>
        <tbody>${plants.map(p => `<tr>
            <td><strong>${p.name}</strong></td>
            <td>${p.location}</td>
            <td class="font-mono">${p.stations}</td>
            <td><span class="instr-badge ${p.maturity === 'High' ? 'full' : p.maturity === 'Medium' ? 'partial' : 'manual'}">${p.maturity}</span></td>
            <td class="font-mono">${p.coverage}</td>
            <td class="text-success font-mono">${p.savings}</td>
            <td>${p.status}</td>
        </tr>`).join('')}</tbody>
    </table>`;
}

// ═══════════════════════════════════════════════════════════════════════════════
// WHAT-IF SIMULATION
// ═══════════════════════════════════════════════════════════════════════════════

async function runWhatIf() {
    const stationSelect = document.getElementById('whatif-station');
    const interventionSelect = document.getElementById('whatif-intervention');
    const resultsContainer = document.getElementById('whatif-results');

    if (!stationSelect || !interventionSelect) return;

    const stationId = parseInt(stationSelect.value);
    const intervention = interventionSelect.value;

    // Build intervention based on type
    let changes = {};
    if (intervention === 'maintenance') {
        changes = { 4: 1.0, 5: 0.5, 10: 0.0 };  // Reset vibration, torque, maint hours
    } else if (intervention === 'takt_increase') {
        changes = { 1: config.stations[stationId].takt * 0.85 };  // Reduce cycle time
    } else if (intervention === 'reroute') {
        changes = { 0: 1.0, 6: 95.0 };  // Reduce queue, increase utilization
    }

    const body = { interventions: { [stationId]: changes } };
    const res = await fetch('/api/whatif', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const data = await res.json();

    // Render results
    const baseline = data.baseline;
    const whatif = data.whatif;

    let html = '<div class="grid-2 mt-md">';
    html += '<div><h4 style="font-size:0.85rem;margin-bottom:8px;color:var(--risk-high)">⬤ Current State</h4>';
    html += renderWhatIfColumn(baseline, stationId);
    html += '</div>';
    html += '<div><h4 style="font-size:0.85rem;margin-bottom:8px;color:var(--risk-low)">⬤ After Intervention</h4>';
    html += renderWhatIfColumn(whatif, stationId);
    html += '</div>';
    html += '</div>';

    // Summary
    const queueReduction = ((baseline.queue[stationId] - whatif.queue[stationId]) / baseline.queue[stationId] * 100);
    html += `<div class="alert-card low mt-md">
        <span class="alert-icon">✅</span>
        <div>
            <div class="alert-title">Intervention Impact</div>
            <div class="alert-detail">
                Queue at ${config.stations[stationId].name}: <strong>${queueReduction.toFixed(0)}% reduction</strong><br>
                Risk level: ${RISK_LABELS[baseline.risk_class[stationId]]} → ${RISK_LABELS[whatif.risk_class[stationId]]}
            </div>
        </div>
    </div>`;

    resultsContainer.innerHTML = html;
}

function renderWhatIfColumn(data, targetStation) {
    // Show target station + 3 downstream
    const stationsToShow = [targetStation];
    config.edges
        .filter(e => e.source === targetStation && e.type === 'forward')
        .forEach(e => stationsToShow.push(e.target));

    let html = '';
    stationsToShow.slice(0, 5).forEach(sid => {
        const s = config.stations[sid];
        const risk = data.risk_class[sid];
        const riskLabel = RISK_LABELS[risk].toLowerCase();
        html += `<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;font-size:0.8rem">
            <span>${s.short}</span>
            <span><span class="risk-badge ${riskLabel}">${RISK_LABELS[risk]}</span> Q:${data.queue[sid].toFixed(1)}</span>
        </div>`;
    });
    return html;
}
