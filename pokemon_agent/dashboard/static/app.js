(() => {
    'use strict';

    const POLL_INTERVAL = 3000;
    const HISTORY_LIMIT = 160;
    const WS_RECONNECT_BASE = 1000;
    const WS_RECONNECT_MAX = 30000;

    const $ = (id) => document.getElementById(id);

    const els = {
        statusDot: $('statusDot'),
        statusText: $('statusText'),
        lastUpdate: $('lastUpdate'),
        uiModeChip: $('uiModeChip'),
        frameTimestamp: $('frameTimestamp'),
        screenTextSource: $('screenTextSource'),
        annotatedFrame: $('annotatedFrame'),
        rawFrame: $('rawFrame'),
        screenText: $('screenText'),
        objectiveTitle: $('objectiveTitle'),
        objectiveProgress: $('objectiveProgress'),
        objectiveSummary: $('objectiveSummary'),
        objectivePredicate: $('objectivePredicate'),
        objectiveRoute: $('objectiveRoute'),
        progressFill: $('progressFill'),
        turnPlanSummary: $('turnPlanSummary'),
        plannedActions: $('plannedActions'),
        fallbackActions: $('fallbackActions'),
        turnPlanNotes: $('turnPlanNotes'),
        recentActionSummary: $('recentActionSummary'),
        recentActionNotes: $('recentActionNotes'),
        stateDeltaSummary: $('stateDeltaSummary'),
        worldStats: $('worldStats'),
        interactionProbe: $('interactionProbe'),
        partySnapshot: $('partySnapshot'),
        liveAscii: $('liveAscii'),
        exploredAscii: $('exploredAscii'),
        checkpointList: $('checkpointList'),
        recoveryRecommendation: $('recoveryRecommendation'),
        recoveryCandidates: $('recoveryCandidates'),
        stuckSignal: $('stuckSignal'),
        knowledgeSummary: $('knowledgeSummary'),
        workspaceSummary: $('workspaceSummary'),
        timeline: $('timeline'),
        rawObservation: $('rawObservation'),
        rawNavigation: $('rawNavigation'),
    };

    let ws = null;
    let wsReconnectDelay = WS_RECONNECT_BASE;
    let wsReconnectTimer = null;
    let pollTimer = null;

    function api(path) {
        return `${window.location.protocol}//${window.location.host}${path}`;
    }

    function wsUrl() {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${proto}//${window.location.host}/ws`;
    }

    function setStatus(connected, label) {
        els.statusDot.classList.toggle('connected', connected);
        els.statusText.textContent = label;
    }

    function formatJSON(value) {
        return JSON.stringify(value ?? {}, null, 2);
    }

    function timeLabel(value) {
        if (!value) return 'No timestamp';
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return value;
        return date.toLocaleString();
    }

    function renderList(node, items, fallback) {
        node.innerHTML = '';
        const list = Array.isArray(items) ? items : [];
        if (!list.length) {
            const li = document.createElement('li');
            li.textContent = fallback;
            node.appendChild(li);
            return;
        }
        list.forEach((item) => {
            const li = document.createElement('li');
            li.textContent = typeof item === 'string' ? item : JSON.stringify(item);
            node.appendChild(li);
        });
    }

    function renderWorldStats(worldState, progress) {
        els.worldStats.innerHTML = '';
        const map = worldState.map || {};
        const player = worldState.player || {};
        const pos = player.position || {};
        const battle = worldState.battle || {};
        const stats = [
            ['Map', map.map_name || 'Unknown'],
            ['Coords', `${pos.x ?? '--'}, ${pos.y ?? '--'}`],
            ['Facing', player.facing || 'unknown'],
            ['Battle', battle.in_battle ? (battle.type || 'active') : 'no'],
            ['Progress', `${progress ?? 0}%`],
        ];
        stats.forEach(([label, value]) => {
            const card = document.createElement('div');
            card.className = 'stat-card';
            card.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
            els.worldStats.appendChild(card);
        });
    }

    function renderParty(party) {
        if (!Array.isArray(party) || !party.length) {
            els.partySnapshot.textContent = 'No party data yet.';
            return;
        }
        const lines = party.map((mon) => {
            const moves = (mon.moves || []).map((move) => move.name || move).join(', ');
            return [
                `${mon.nickname || mon.species || 'Unknown'} Lv${mon.level || '?'} ${mon.hp || '?'} / ${mon.max_hp || '?'}`,
                `Status: ${mon.status || 'OK'} | Moves: ${moves || 'none'}`,
            ].join('\n');
        });
        els.partySnapshot.textContent = lines.join('\n\n');
    }

    function renderTimeline(events) {
        els.timeline.innerHTML = '';
        const recent = Array.isArray(events) ? events : [];
        if (!recent.length) {
            els.timeline.innerHTML = '<p class="empty">No events recorded yet.</p>';
            return;
        }
        recent.slice().reverse().forEach((event) => {
            const article = document.createElement('article');
            article.className = 'timeline-item';
            const summary = event.summary || event.reason || event.text || event.objective?.title || event.type;
            article.innerHTML = `
                <div class="timeline-head">
                    <span class="timeline-type">${event.type}</span>
                    <span class="timeline-time">${timeLabel(event.timestamp)}</span>
                </div>
                <p>${summary}</p>
            `;
            els.timeline.appendChild(article);
        });
    }

    function renderDashboardState(payload) {
        const visuals = payload.visuals || {};
        const intent = payload.agent_intent || {};
        const world = payload.world_state || {};
        const memory = payload.memory_and_progress || {};
        const objective = intent.objective || {};
        const turnPlan = intent.turn_plan || {};
        const recentAction = intent.recent_action || {};
        const stateDelta = intent.state_delta || {};
        const screenText = visuals.screen_text || {};
        const recovery = memory.recovery || {};
        const workspace = memory.workspace || {};

        setStatus(true, 'Connected');
        els.lastUpdate.textContent = timeLabel(payload.generated_at);
        els.uiModeChip.textContent = `UI: ${visuals.ui_mode || 'unknown'}`;
        els.frameTimestamp.textContent = timeLabel(visuals.frame_timestamp);
        els.screenTextSource.textContent = `OCR: ${screenText.source || 'n/a'}`;

        if (visuals.annotated_frame_b64) {
            els.annotatedFrame.src = `data:image/png;base64,${visuals.annotated_frame_b64}`;
        }
        if (visuals.raw_frame_b64) {
            els.rawFrame.src = `data:image/png;base64,${visuals.raw_frame_b64}`;
        }
        els.screenText.textContent = screenText.text || 'No OCR or dialogue text available.';

        els.objectiveTitle.textContent = objective.title || 'No objective yet';
        els.objectiveProgress.textContent = `${objective.progress_percent ?? payload.memory_and_progress?.progress_percent ?? 0}%`;
        els.objectiveSummary.textContent = objective.summary || 'No objective summary.';
        els.objectivePredicate.textContent = objective.completion_predicate || '';
        els.objectiveRoute.textContent = objective.route_hint || '';
        els.progressFill.style.width = `${payload.memory_and_progress?.progress_percent ?? 0}%`;

        els.turnPlanSummary.textContent = turnPlan.summary || 'No turn plan summary.';
        renderList(els.plannedActions, turnPlan.planned_actions, 'No planned actions set.');
        renderList(els.fallbackActions, turnPlan.fallback_actions, 'No fallback actions set.');
        els.turnPlanNotes.textContent = turnPlan.notes || `Plan updated: ${turnPlan.updated_at || 'never'}`;

        els.recentActionSummary.textContent = recentAction.summary || 'No recent action summary.';
        renderList(els.recentActionNotes, recentAction.notes, 'No recent action notes.');
        renderList(els.stateDeltaSummary, stateDelta.summary, 'No state delta summary.');

        renderWorldStats(world, memory.progress_percent);
        els.interactionProbe.textContent = formatJSON(world.interaction || {});
        renderParty(world.party || []);
        els.liveAscii.textContent = world.live_ascii || 'No live navigation ASCII available.';
        els.exploredAscii.textContent = world.explored_ascii || 'No explored map ASCII available.';

        renderList(
            els.checkpointList,
            (memory.checkpoints || []).map((checkpoint) => {
                const title = checkpoint.title || checkpoint.id || 'checkpoint';
                return `${title} (${timeLabel(checkpoint.created_at || checkpoint.timestamp)})`;
            }),
            'No checkpoints recorded yet.'
        );

        const recommendation = recovery.current_recommendation || {};
        els.recoveryRecommendation.textContent = recommendation.name
            ? `${recommendation.name} (${recommendation.reason})`
            : 'No recovery recommendation yet.';
        renderList(
            els.recoveryCandidates,
            (recovery.candidates || []).map((candidate) => {
                return `${candidate.name} | ${candidate.reason} | score ${candidate.score}`;
            }),
            'No recovery candidates available.'
        );
        if (memory.stuck) {
            els.stuckSignal.textContent = `${memory.stuck.level}: ${memory.stuck.reason}`;
        } else {
            els.stuckSignal.textContent = 'No stuck signal yet.';
        }
        els.knowledgeSummary.textContent = formatJSON(memory.knowledge_graph_summary || {});
        els.workspaceSummary.textContent = formatJSON(workspace);

        els.rawObservation.textContent = formatJSON(payload.raw || {});
        els.rawNavigation.textContent = formatJSON(world.navigation || {});
    }

    async function fetchDashboardState() {
        const response = await fetch(api('/dashboard/state'));
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        renderDashboardState(payload);
        return payload;
    }

    async function fetchDashboardHistory() {
        const response = await fetch(api(`/dashboard/history?limit=${HISTORY_LIMIT}`));
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        renderTimeline(payload.events || []);
    }

    async function refreshAll() {
        try {
            await Promise.all([fetchDashboardState(), fetchDashboardHistory()]);
        } catch (error) {
            setStatus(false, 'Server unavailable');
        }
    }

    function scheduleReconnect() {
        if (wsReconnectTimer) return;
        wsReconnectTimer = setTimeout(() => {
            wsReconnectTimer = null;
            connectWS();
        }, wsReconnectDelay);
        wsReconnectDelay = Math.min(wsReconnectDelay * 2, WS_RECONNECT_MAX);
    }

    function connectWS() {
        if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
            return;
        }
        try {
            ws = new WebSocket(wsUrl());
        } catch (_) {
            scheduleReconnect();
            return;
        }

        ws.onopen = () => {
            wsReconnectDelay = WS_RECONNECT_BASE;
            setStatus(true, 'Connected');
            refreshAll();
        };

        ws.onmessage = () => {
            refreshAll();
        };

        ws.onclose = () => {
            setStatus(false, 'Disconnected');
            scheduleReconnect();
        };

        ws.onerror = () => {
            setStatus(false, 'Error');
        };
    }

    function init() {
        setStatus(false, 'Connecting');
        refreshAll();
        connectWS();
        pollTimer = setInterval(refreshAll, POLL_INTERVAL);
        window.addEventListener('beforeunload', () => {
            if (pollTimer) window.clearInterval(pollTimer);
            if (wsReconnectTimer) window.clearTimeout(wsReconnectTimer);
            if (ws) ws.close();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
