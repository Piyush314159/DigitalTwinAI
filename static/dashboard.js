/* DigitalTwin operations. All operational status comes from the frozen detector.
   Replay requests are serialized; charts stop at the last observed step. */
'use strict';

let config = null;
let state = null;
let evaluation = null;
let selectedStation = 7;
let activeView = 'overview';
let busy = false;
let playing = false;
let replayTimer = null;
let scoreDescending = true;
const $ = id => document.getElementById(id);
const icon = name => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fixed = (value, digits = 3) => value === null || value === undefined ? '—' : Number(value).toFixed(digits);
const number = value => value === null || value === undefined ? '—' : Number(value).toLocaleString('en-US');
const ready = () => state?.modelStatus === 'trained_synthetic';
const zones = ['Body Construction', 'Paint Shop', 'Final Assembly'];
const pages = {
    overview: ['OPERATIONS / OVERVIEW', 'Assembly line overview', 'Current conditions across 40 stations in three production zones.'],
    stations: ['OPERATIONS / STATIONS', 'Station register', 'Observed readings, detection scores and sensor availability.'],
    validation: ['MODEL / EVALUATION', 'Model validation', 'Held-out performance and six frozen-model robustness conditions.'],
    methodology: ['MODEL / METHODOLOGY', 'Model methodology', 'Data sources, evaluation protocol and capability status.'],
};

async function api(path, options) {
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The local service could not complete this request.');
    return data;
}

function showError(message) {
    $('error-banner').textContent = message;
    $('error-banner').hidden = false;
}

function changeView(view) {
    if (!pages[view]) return;
    activeView = view;
    const [eyebrow, title, description] = pages[view];
    $('page-eyebrow').textContent = eyebrow;
    $('page-title').textContent = title;
    $('page-description').textContent = description;
    document.querySelectorAll('.view').forEach(element => { element.hidden = element.id !== `view-${view}`; });
    document.querySelectorAll('.nav-item').forEach(button => {
        const active = button.dataset.view === view;
        button.classList.toggle('active', active);
        if (active) button.setAttribute('aria-current', 'page');
        else button.removeAttribute('aria-current');
    });
    $('replay-controls').hidden = view === 'validation' || view === 'methodology';
    if (view === 'validation' || view === 'methodology') stopReplay();
    const exportEvaluation = view === 'validation' || view === 'methodology';
    $('export-link').href = exportEvaluation ? '/api/evaluation.json' : '/api/snapshot.csv';
    $('export-label').textContent = exportEvaluation ? 'Export evaluation' : 'Export snapshot';
    history.replaceState(null, '', `#${view}`);
    window.scrollTo(0, 0);
    document.title = `${view === 'overview' ? 'Operations' : title.replace(/[.,]$/, '')} — DigitalTwin`;
}

function setBusy(value) {
    busy = value;
    updateControls();
}

function updateControls() {
    const ended = state && state.step === state.totalSteps - 1;
    $('btn-reset').disabled = busy || !state;
    $('btn-step').disabled = busy || !state || ended;
    $('btn-play').disabled = busy || !state || (ended && !playing);
    $('step-slider').disabled = busy || !state;
    $('btn-play').innerHTML = icon(playing ? 'pause' : 'play') + `<span>${playing ? 'Pause' : 'Play shift'}</span>`;
    if (!state) return;
    $('step-slider').max = state.totalSteps - 1;
    $('step-slider').value = state.step;
    $('step-label').textContent = `${String(state.step).padStart(2, '0')} / ${state.totalSteps - 1}`;
    $('replay-time').textContent = state.timestamp;
}

function stopReplay() {
    playing = false;
    clearTimeout(replayTimer);
    replayTimer = null;
    updateControls();
}

async function mutate(path, body) {
    if (busy) return false;
    setBusy(true);
    try {
        state = await api(path, {method: 'POST', ...(body ? {headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {})});
        $('error-banner').hidden = true;
        if (state.step === state.totalSteps - 1) stopReplay();
        render();
        $('live-announcement').textContent = `Step ${state.step}, ${state.summary.flaggedStations ?? 'unavailable'} flagged stations.`;
        return true;
    } catch (error) {
        stopReplay();
        showError(`${error.message} The last loaded observations are still shown.`);
        return false;
    } finally {
        setBusy(false);
    }
}

async function replayNext() {
    if (!playing) return;
    const success = await mutate('/api/step');
    if (playing && success) replayTimer = setTimeout(replayNext, 1000);
}

function metric(label, value, note, unit = '', alert = false) {
    return `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value ${alert ? 'is-alert' : ''}">${value}${unit ? `<small>${unit}</small>` : ''}</div><div class="metric-note">${note}</div></div>`;
}

function renderMetrics() {
    const s = state.summary;
    $('overview-metrics').innerHTML =
        metric('Stations monitored', number(s.observedStations), 'Across 3 production zones', '/ 40') +
        metric('Stations flagged', number(s.flaggedStations), ready() ? `Detection threshold ${fixed(state.threshold, 2)}` : 'Detector unavailable', '', s.flaggedStations > 0) +
        metric('Sensor coverage', fixed(s.sensorCoverage * 100, 1), 'Observed readings / all features', '%') +
        metric('Mean observed queue', fixed(s.meanObservedQueue, 1), 'Current reading · not a forecast', 'units');
    $('sidebar-model-state').textContent = ready() ? 'Anomaly detector loaded' : 'Detector unavailable';
    $('sidebar-model-dot').style.background = ready() ? '#a4bf8d' : '#9b8f7e';
}

function nodeStatus(id) {
    return !ready() ? 'unavailable' : state.flagged[id] ? 'flagged' : 'below';
}

function statusChip(id) {
    const status = nodeStatus(id);
    const text = status === 'flagged' ? 'Flagged' : status === 'below' ? 'Below threshold' : 'Unavailable';
    return `<span class="status-chip ${status === 'flagged' ? 'flagged' : status === 'unavailable' ? 'unavailable' : ''}"><i class="dot ${status === 'flagged' ? 'dot-flagged' : status === 'unavailable' ? 'dot-unavailable' : 'dot-normal'}"></i>${text}</span>`;
}

function selectStation(id) {
    if (!config || !config.stations.some(s => s.id === id)) return;
    selectedStation = id;
    renderMap();
    renderAlerts();
    renderStationDetails();
    renderStationTable();
}

function renderMap() {
    const positions = new Map();
    let bands = '';
    zones.forEach((zone, zoneIndex) => {
        const group = config.stations.filter(s => s.zone === zone);
        const top = 10 + zoneIndex * 110;
        bands += `<rect x="10" y="${top}" width="740" height="102" rx="4" fill="${zoneIndex % 2 ? '#f8faf6' : '#f2f5ef'}"/><text x="26" y="${top + 18}" class="zone-title">0${zoneIndex + 1} / ${zone.toUpperCase()}<tspan fill="#9aa695"> · ${group.length} STATIONS</tspan></text>`;
        group.forEach((station, local) => {
            const column = local < 8 ? local : 15 - local;
            positions.set(station.id, [82 + column * 85, top + (local < 8 ? 47 : 82)]);
        });
    });
    const feedback = $('show-feedback').checked;
    let edges = '';
    config.edges.forEach(edge => {
        if (edge.type !== 'forward' && !feedback) return;
        const [sx, sy] = positions.get(edge.source), [tx, ty] = positions.get(edge.target);
        const backwards = edge.type !== 'forward';
        let path;
        if (backwards) path = `M${sx},${sy} Q${(sx + tx) / 2},${Math.min(sy, ty) - 38} ${tx},${ty}`;
        else if (sy === ty) path = `M${sx},${sy} L${tx},${ty}`;
        else path = `M${sx},${sy} C${sx},${sy + 28} ${tx},${ty - 28} ${tx},${ty}`;
        edges += `<path d="${path}" fill="none" stroke="${backwards ? '#a0b298' : '#cad5c5'}" stroke-width="1.1" ${backwards ? 'stroke-dasharray="3 4"' : ''} marker-end="url(#flow-arrow)"/>`;
    });
    let nodes = '';
    config.stations.forEach(station => {
        const [x, y] = positions.get(station.id);
        const status = nodeStatus(station.id);
        const selected = station.id === selectedStation;
        const fill = status === 'flagged' ? '#f6e3d5' : status === 'unavailable' ? '#e6e8e1' : '#e0eadb';
        const stroke = status === 'flagged' ? '#c57b57' : status === 'unavailable' ? '#afb8ab' : '#91aa84';
        const color = status === 'flagged' ? '#8d482b' : '#48603f';
        const score = ready() ? fixed(state.scores[station.id]) : 'unavailable';
        nodes += `<g class="station-node" role="button" tabindex="0" data-station="${station.id}" aria-label="Station ${String(station.id).padStart(2, '0')}: ${escapeHTML(station.name)}, ${status === 'below' ? 'below threshold' : status}" aria-pressed="${selected}">
            <title>${escapeHTML(station.name)} · score ${score}</title>
            ${selected ? `<circle cx="${x}" cy="${y}" r="16" fill="#fff" stroke="#315b40" stroke-width="1.4"/>` : ''}
            <circle class="node-face" cx="${x}" cy="${y}" r="12" fill="${fill}" stroke="${stroke}"/>
            <text x="${x}" y="${y}" class="node-number" fill="${color}">${String(station.id).padStart(2, '0')}</text></g>`;
    });
    $('flow-map').innerHTML = `<svg class="flow-svg" viewBox="0 0 760 340" xmlns="http://www.w3.org/2000/svg" role="group" aria-label="Forty stations across three production zones"><defs><marker id="flow-arrow" viewBox="0 0 8 8" refX="17" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 1 7 4 0 7" fill="none" stroke="#aabda3" stroke-width="1"/></marker></defs>${bands}${edges}${nodes}</svg>`;
    $('flow-caption').textContent = `42 forward links · 14 feedback links ${feedback ? 'shown' : 'hidden'}`;
}

function renderAlerts() {
    $('alert-count').textContent = ready() ? state.summary.flaggedStations : '—';
    if (!ready()) {
        $('alerts-list').innerHTML = `<div class="empty-alerts">${icon('circle')}<strong>Detection unavailable</strong><p>Restore the trained checkpoint to evaluate station conditions.</p></div>`;
        return;
    }
    const alerts = config.stations.filter(s => state.flagged[s.id]).sort((a, b) => state.scores[b.id] - state.scores[a.id]);
    if (!alerts.length) {
        $('alerts-list').innerHTML = `<div class="empty-alerts">${icon('validation')}<strong>No stations flagged</strong><p>All detection scores are below ${fixed(state.threshold, 2)} at this observed step.</p></div>`;
        return;
    }
    $('alerts-list').innerHTML = alerts.map(s => `<button type="button" class="alert-row ${s.id === selectedStation ? 'selected' : ''}" data-station="${s.id}" aria-label="Inspect ${escapeHTML(s.name)}"><span class="alert-id">${String(s.id).padStart(2, '0')}</span><span class="alert-text"><strong>${escapeHTML(s.name)}</strong><small>${escapeHTML(s.zone)}</small></span><span class="alert-score">${fixed(state.scores[s.id])}</span>${icon('arrow')}</button>`).join('');
}

function renderActivity() {
    const width = 650, left = 28, right = 12, top = 12, bottom = 142;
    const maxCount = Math.max(4, ...state.history.map(p => p.flaggedCount || 0));
    const ceiling = Math.ceil(maxCount / 2) * 2;
    const x = step => left + step / (state.totalSteps - 1) * (width - left - right);
    const y = count => bottom - count / ceiling * (bottom - top);
    let grid = '';
    [0, ceiling / 2, ceiling].forEach(value => {
        grid += `<line x1="${left}" x2="${width - right}" y1="${y(value)}" y2="${y(value)}" stroke="#edf0e9"/><text x="${left - 9}" y="${y(value) + 3}" class="chart-label" text-anchor="end">${value}</text>`;
    });
    [0, 10, 20, 29].forEach(step => {grid += `<text x="${x(step)}" y="164" class="chart-label" text-anchor="${step === 0 ? 'start' : step === 29 ? 'end' : 'middle'}">${step === 0 ? '08:00' : step === 10 ? '10:40' : step === 20 ? '13:20' : '15:44'}</text>`;});
    let plot = '';
    if (ready()) {
        const points = state.history.map(p => [x(p.step), y(p.flaggedCount)]);
        const line = points.map(([px, py], i) => `${i ? 'L' : 'M'}${px.toFixed(2)},${py.toFixed(2)}`).join(' ');
        const area = `${line} L${x(state.step)},${bottom} L${left},${bottom} Z`;
        plot = `<path d="${area}" fill="#eaf0e6"/><path d="${line}" stroke="#527a4f" stroke-width="2" stroke-linejoin="round" fill="none"/><circle cx="${x(state.step)}" cy="${y(state.summary.flaggedStations)}" r="3.5" fill="#3d6742" stroke="white" stroke-width="1.5"/>`;
    }
    $('activity-chart').innerHTML = `<svg viewBox="0 0 650 176" role="img" aria-label="Flagged station counts through step ${state.step}; future steps are unobserved">${state.step < 29 ? `<rect x="${x(state.step)}" y="${top}" width="${width - right - x(state.step)}" height="${bottom - top}" fill="#fafbf8"/>` : ''}${grid}${plot}${!ready() ? '<text x="325" y="78" text-anchor="middle" class="chart-label">Detector unavailable</text>' : state.step < 18 ? `<text x="${(x(state.step) + width - right) / 2}" y="70" text-anchor="middle" class="chart-label">Unobserved</text>` : ''}</svg>`;
    $('activity-caption').textContent = `${state.step + 1} / ${state.totalSteps} steps observed`;
}

function readingHTML(value, index) {
    if (value === null || value === undefined) return '<dd class="missing">Not observed</dd>';
    const feature = config.features[index];
    return `<dd>${fixed(value * feature.multiplier, feature.decimals)}${feature.unit ? `<small>${escapeHTML(feature.unit)}</small>` : ''}</dd>`;
}

function scoreSparkline(id) {
    if (!ready()) return '<p>Detection history unavailable.</p>';
    const points = state.history.map(p => `${10 + p.step / 29 * 220},${75 - p.scores[id] * 65}`).join(' ');
    return `<p>Detection score / observed shift</p><svg viewBox="0 0 240 90" role="img" aria-label="Station detection score history with fixed threshold ${fixed(state.threshold, 2)}"><line x1="10" x2="230" y1="${75 - state.threshold * 65}" y2="${75 - state.threshold * 65}" stroke="#d4b29a" stroke-dasharray="3 3"/><polyline points="${points}" fill="none" stroke="#456d48" stroke-width="1.7"/><text x="230" y="88" text-anchor="end" class="chart-label">Threshold ${fixed(state.threshold, 2)}</text></svg>`;
}

function renderStationDetails() {
    const station = config.stations.find(s => s.id === selectedStation);
    const values = state.observations[selectedStation];
    const base = `<div class="panel-heading"><div><div class="station-code">STATION ${String(station.id).padStart(2, '0')} / ${escapeHTML(station.zone.toUpperCase())}</div><h2 class="station-heading">${escapeHTML(station.name)}</h2></div></div>`;
    const score = `<div class="selected-score"><span>Detection score <strong>${ready() ? fixed(state.scores[station.id]) : '—'}</strong></span>${statusChip(station.id)}</div>`;
    const readings = indices => `<dl class="reading-grid">${indices.map(index => `<div class="reading"><dt>${escapeHTML(config.features[index].name)}</dt>${readingHTML(values[index], index)}</div>`).join('')}</dl>`;
    $('station-detail').innerHTML = base + readings([0, 1, 2, 3, 4, 6]) + score;
    $('station-detail-secondary').innerHTML = base + readings(config.features.map((_, i) => i)) + score + `<div class="station-history">${scoreSparkline(station.id)}</div><div class="panel-foot"><span>${values.filter(v => v !== null).length} / 12 readings observed</span><span>${escapeHTML(station.instrumentation)} sensors</span></div>`;
}

function renderStationTable() {
    if (!state || !config) return;
    const query = $('station-search').value.trim().toLowerCase();
    const zone = $('zone-filter').value;
    const status = $('status-filter').value;
    const rows = config.stations.filter(s =>
        (!query || `${s.name} ${s.short} ${String(s.id).padStart(2, '0')}`.toLowerCase().includes(query)) &&
        (zone === 'all' || s.zone === zone) && (status === 'all' || nodeStatus(s.id) === status));
    rows.sort((a, b) => ready() ? (scoreDescending ? -1 : 1) * (state.scores[a.id] - state.scores[b.id]) || a.id - b.id : a.id - b.id);
    $('station-result-count').textContent = `${rows.length} of 40 stations · ${ready() ? 'sorted by detection score' : 'detector unavailable'}`;
    $('station-table-body').innerHTML = rows.length ? rows.map(s => `<tr class="${s.id === selectedStation ? 'is-selected' : ''}"><td><button type="button" class="table-station" data-station="${s.id}"><span class="mono">${String(s.id).padStart(2, '0')}</span><strong>${escapeHTML(s.name)}</strong></button></td><td>${s.zone === 'Body Construction' ? 'Body' : s.zone === 'Paint Shop' ? 'Paint' : 'Final'}</td><td>${statusChip(s.id)}</td><td class="score-cell ${nodeStatus(s.id) === 'flagged' ? 'flagged' : ''}">${ready() ? fixed(state.scores[s.id]) : '—'}</td><td class="mono">${fixed(state.observations[s.id][0], 1)}</td><td class="mono">${state.observations[s.id].filter(v => v !== null).length} / 12</td><td><button type="button" class="table-inspect" data-station="${s.id}" aria-label="Inspect readings for ${escapeHTML(s.name)}">${icon('arrow')}</button></td></tr>`).join('') : '<tr><td colspan="7" class="table-empty">No stations match these filters.</td></tr>';
}

function renderValidation() {
    const container = $('validation-content');
    if (!evaluation?.metrics) {
        container.innerHTML = '<div class="panel prose-panel"><h2>Evaluation unavailable</h2><p>A matching saved model and evaluation report are required to display measured results.</p></div>';
        return;
    }
    const m = evaluation.metrics, baseline = evaluation.baseline;
    const cases = evaluation.robustness?.cases;
    const comparison = (label, value, isBaseline = false) => `<div class="comparison-row"><div class="comparison-label"><span>${label}</span><strong>${fixed(value)}</strong></div><div class="comparison-track"><div class="comparison-fill ${isBaseline ? 'baseline' : ''}" style="width:${value * 100}%"></div></div></div>`;
    container.innerHTML = `<div class="validation-intro">${icon('validation')}<p>Current-station anomaly detection. Results below come from separate simulated shifts and match the loaded checkpoint. All splits share the same simulator and graph; real-factory performance is unestablished.</p></div>
        <div class="metrics-strip validation-metrics">${metric('Held-out F1', fixed(m.f1), 'Balance of precision and recall')}${metric('Precision', fixed(m.precision), `${number(m.false_positive)} false-positive station-steps`)}${metric('Recall', fixed(m.recall), `${number(m.false_negative)} missed station-steps`)}${metric('Average precision', fixed(m.average_precision), 'Ranking metric · not accuracy')}</div>
        <div class="validation-grid"><section class="panel"><div class="panel-heading"><div><h2>Compared with a simple baseline</h2><p>Original held-out F1 · fixed validation-selected thresholds</p></div></div><div class="comparison-body">${comparison('GAT + GRU detector', m.f1)}${comparison('Local linear baseline', baseline.f1, true)}<p class="comparison-note">${number(evaluation.parameters)} parameters · ${evaluation.split_counts.train} training / ${evaluation.split_counts.validation} validation / ${evaluation.split_counts.test} test shifts.<br>This comparison does not isolate graph structure from recurrence or nonlinear model capacity.</p></div></section>
        <section class="panel"><div class="panel-heading"><div><h2>What the detector got right—and missed</h2><p>${number(m.observations)} correlated station-time observations</p></div></div><div class="confusion-grid"><div class="confusion-cell"><strong>${number(m.true_positive)}</strong><span>Disturbances detected</span></div><div class="confusion-cell is-error"><strong>${number(m.false_negative)}</strong><span>Disturbances missed</span></div><div class="confusion-cell is-error"><strong>${number(m.false_positive)}</strong><span>False alarms</span></div><div class="confusion-cell"><strong>${number(m.true_negative)}</strong><span>Normal station-steps below threshold</span></div></div></section>
        <section class="panel robustness-panel"><div class="panel-heading"><div><h2>Robustness under harder conditions</h2><p>Saved model and threshold ${fixed(state.threshold, 2)} remain fixed throughout evaluation</p></div><span class="label-pill">${cases ? '6 conditions' : 'Not evaluated'}</span></div>
        ${cases ? `<div class="table-scroll"><table class="data-table robustness-table"><thead><tr><th>Condition</th><th>F1</th><th>Recall</th><th>False alarms</th><th>Misses</th></tr></thead><tbody>${cases.map(c => `<tr class="${c.id === 'sensor_outages' ? 'degraded' : ''}"><td><div class="condition-name">${escapeHTML(c.title)}</div><div class="condition-note">${escapeHTML(c.description)}</div></td><td class="mono">${fixed(c.detector.f1)}</td><td class="mono">${fixed(c.detector.recall)}</td><td class="mono">${number(c.detector.false_positive)}</td><td class="mono">${number(c.detector.false_negative)}</td></tr>`).join('')}</tbody></table></div><div class="robustness-foot">${cases[0].shifts} complete shifts per condition. Sensor outage and noise cases share the fresh reference shifts for paired comparisons. F1 and recall are undefined for all-healthy shifts. Counts are station-steps, not independent factory incidents.<br>Additional sensor outages expose the largest observed degradation. These inspected challenge sets should not be reused for final model tuning.</div>` : '<div class="prose-panel"><p class="small-note">Run the declared robustness evaluation to inspect additional conditions.</p></div>'}</section></div>`;
    $('footer-model').textContent = `CHECKPOINT ${evaluation.checkpoint_sha256.slice(0, 8).toUpperCase()} / SYNTHETIC`;
}

function renderCapabilities() {
    $('capability-list').innerHTML = config.capabilities.map(capability => `<div class="capability-row"><span>${escapeHTML(capability.name)}</span><span class="${capability.status === 'trained_synthetic' ? 'available' : ''}">${capability.status === 'trained_synthetic' ? 'Trained · synthetic' : capability.status === 'not_trained' ? 'Not trained' : 'Not validated'}</span></div>`).join('');
}

function render() {
    if (!config || !state) return;
    updateControls();
    renderMetrics();
    renderMap();
    renderAlerts();
    renderActivity();
    renderStationDetails();
    renderStationTable();
}

document.addEventListener('DOMContentLoaded', async () => {
    setBusy(true);
    try {
        config = await api('/api/config');
        $('zone-filter').insertAdjacentHTML('beforeend', zones.map(zone => `<option>${escapeHTML(zone)}</option>`).join(''));
        [state, evaluation] = await Promise.all([api('/api/state'), api('/api/validation')]);
        render();
        renderValidation();
        renderCapabilities();
        changeView(pages[location.hash.slice(1)] ? location.hash.slice(1) : 'overview');
    } catch (error) {
        showError(`${error.message} Check that the local service is running, then reload.`);
    } finally {
        setBusy(false);
    }
});

document.addEventListener('click', event => {
    const viewButton = event.target.closest('[data-view]');
    if (viewButton) { changeView(viewButton.dataset.view); return; }
    const station = event.target.closest('[data-station]');
    if (station) selectStation(Number(station.dataset.station));
});
document.addEventListener('keydown', event => {
    const station = event.target.closest('.station-node');
    if (station && (event.key === 'Enter' || event.key === ' ')) {event.preventDefault(); selectStation(Number(station.dataset.station));}
});
$('btn-step').addEventListener('click', () => {stopReplay(); mutate('/api/step');});
$('btn-reset').addEventListener('click', () => {stopReplay(); mutate('/api/reset');});
$('btn-play').addEventListener('click', () => {
    if (playing) stopReplay();
    else {playing = true; updateControls(); replayNext();}
});
$('step-slider').addEventListener('change', event => {
    const requestedStep = Number(event.target.value);
    stopReplay();
    mutate('/api/seek', {step: requestedStep});
});
$('show-feedback').addEventListener('change', renderMap);
$('station-search').addEventListener('input', renderStationTable);
$('zone-filter').addEventListener('change', renderStationTable);
$('status-filter').addEventListener('change', renderStationTable);
$('sort-score').addEventListener('click', () => {
    scoreDescending = !scoreDescending;
    $('sort-score').innerHTML = `Score <span aria-hidden="true">${scoreDescending ? '↓' : '↑'}</span>`;
    renderStationTable();
});
window.addEventListener('hashchange', () => { if (pages[location.hash.slice(1)]) changeView(location.hash.slice(1)); });
