(() => {
    'use strict';

    const POLL_INTERVAL = 3000;
    const HISTORY_LIMIT = 180;
    const TRANSCRIPT_LIMIT = 160;
    const WS_RECONNECT_BASE = 1000;
    const WS_RECONNECT_MAX = 30000;

    const $ = (id) => document.getElementById(id);

    const els = {
        statusDot: $('statusDot'),
        statusText: $('statusText'),
        piStatusChip: $('piStatusChip'),
        piStatusDot: $('piStatusDot'),
        piStatusText: $('piStatusText'),
        lastUpdate: $('lastUpdate'),
        uiModeChip: $('uiModeChip'),
        realtimeChip: $('realtimeChip'),
        piModelChip: $('piModelChip'),
        frameTimestamp: $('frameTimestamp'),
        screenTextSource: $('screenTextSource'),
        annotatedFrame: $('annotatedFrame'),
        rawFrame: $('rawFrame'),
        screenText: $('screenText'),
        piSessionChip: $('piSessionChip'),
        piTurnsChip: $('piTurnsChip'),
        piStatusSummary: $('piStatusSummary'),
        piSupervisorStats: $('piSupervisorStats'),
        piPromptInput: $('piPromptInput'),
        piProviderInput: $('piProviderInput'),
        piModelInput: $('piModelInput'),
        piThinkingSelect: $('piThinkingSelect'),
        piContinueMessageInput: $('piContinueMessageInput'),
        piAutoContinueInput: $('piAutoContinueInput'),
        piStartButton: $('piStartButton'),
        piContinueButton: $('piContinueButton'),
        piStopButton: $('piStopButton'),
        piControlStatus: $('piControlStatus'),
        piTurnPlanPreview: $('piTurnPlanPreview'),
        piActiveTools: $('piActiveTools'),
        piCurrentThinking: $('piCurrentThinking'),
        piCurrentOutput: $('piCurrentOutput'),
        piRecentTools: $('piRecentTools'),
        piTranscript: $('piTranscript'),
        piRecentEvents: $('piRecentEvents'),
        piStderr: $('piStderr'),
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
        movementGuidance: $('movementGuidance'),
        worldStats: $('worldStats'),
        interactionProbe: $('interactionProbe'),
        partySnapshot: $('partySnapshot'),
        liveAscii: $('liveAscii'),
        exploredAscii: $('exploredAscii'),
        checkpointList: $('checkpointList'),
        recoveryRecommendation: $('recoveryRecommendation'),
        recoveryCandidates: $('recoveryCandidates'),
        stuckSignal: $('stuckSignal'),
        manualSaveNameInput: $('manualSaveNameInput'),
        manualSaveButton: $('manualSaveButton'),
        manualSaveStatus: $('manualSaveStatus'),
        saveSelect: $('saveSelect'),
        loadSaveButton: $('loadSaveButton'),
        loadRecommendedButton: $('loadRecommendedButton'),
        loadSaveStatus: $('loadSaveStatus'),
        knowledgeSummary: $('knowledgeSummary'),
        workspaceSummary: $('workspaceSummary'),
        timeline: $('timeline'),
        rawObservation: $('rawObservation'),
        rawNavigation: $('rawNavigation'),
        rawSupervisor: $('rawSupervisor'),
    };

    let ws = null;
    let wsReconnectDelay = WS_RECONNECT_BASE;
    let wsReconnectTimer = null;
    let pollTimer = null;
    let refreshTimer = null;
    let refreshInFlight = null;
    let controlSeeded = false;
    let latestRecovery = {};
    let latestSaves = [];
    const transcriptKeys = new Set();

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

    function setPiStatus(status, label) {
        const normalized = status || 'idle';
        els.piStatusChip.dataset.status = normalized;
        els.piStatusDot.className = 'status-dot';
        if (normalized === 'running' || normalized === 'starting') {
            els.piStatusDot.classList.add('running');
        } else if (normalized === 'error') {
            els.piStatusDot.classList.add('error');
        } else if (normalized === 'stopping') {
            els.piStatusDot.classList.add('warning');
        } else {
            els.piStatusDot.classList.add('idle');
        }
        els.piStatusText.textContent = label;
    }

    function formatJSON(value) {
        return JSON.stringify(value ?? {}, null, 2);
    }

    function withCacheBust(url, token) {
        if (!url) return '';
        const suffix = token ? encodeURIComponent(token) : String(Date.now());
        return `${url}${url.includes('?') ? '&' : '?'}t=${suffix}`;
    }

    function timeLabel(value) {
        if (!value) return 'No timestamp';
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return value;
        return date.toLocaleString();
    }

    function truncate(value, limit = 220) {
        const text = String(value ?? '').trim();
        if (!text) return '';
        if (text.length <= limit) return text;
        return `${text.slice(0, limit - 1).trimEnd()}…`;
    }

    function defaultSaveName() {
        const date = new Date();
        const pad = (value) => String(value).padStart(2, '0');
        const yyyy = date.getFullYear();
        const mm = pad(date.getMonth() + 1);
        const dd = pad(date.getDate());
        const hh = pad(date.getHours());
        const mi = pad(date.getMinutes());
        const ss = pad(date.getSeconds());
        return `manual_${yyyy}${mm}${dd}_${hh}${mi}${ss}`;
    }

    function reasonLabel(reason) {
        return String(reason || 'manual_save').replaceAll('_', ' ');
    }

    function inferredSaveReason(name) {
        if (!name || !name.startsWith('auto__')) return 'manual_save';
        const parts = name.split('__');
        return parts[2] ? parts[2].replaceAll('-', '_') : 'auto_save';
    }

    function modifiedTimeLabel(value) {
        if (typeof value !== 'number') return 'unknown time';
        return timeLabel(new Date(value * 1000).toISOString());
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

    function renderKeyValueCards(node, pairs) {
        node.innerHTML = '';
        pairs.forEach(([label, value]) => {
            const card = document.createElement('div');
            card.className = 'stat-card';
            card.innerHTML = `<span>${label}</span><strong>${value || 'n/a'}</strong>`;
            node.appendChild(card);
        });
    }

    function renderToolList(node, items, fallback) {
        node.innerHTML = '';
        const list = Array.isArray(items) ? items : [];
        if (!list.length) {
            node.innerHTML = `<p class="empty">${fallback}</p>`;
            return;
        }
        list.forEach((item) => {
            const article = document.createElement('article');
            article.className = 'tool-card';
            const header = truncate(item.summary || item.tool_name || 'tool');
            const detail = truncate(item.result_preview || item.args_preview || '');
            article.innerHTML = `
                <div class="tool-head">
                    <strong>${header}</strong>
                    <span class="inline-chip">${item.status || item.tool_name || 'tool'}</span>
                </div>
                <p>${detail || 'No preview available.'}</p>
            `;
            node.appendChild(article);
        });
    }

    function buildRecoveryMap(recovery) {
        const map = new Map();
        const candidates = recovery?.candidates || [];
        candidates.forEach((candidate) => {
            if (candidate?.name) {
                map.set(candidate.name, candidate);
            }
        });
        return map;
    }

    function optionLabelForSave(save, recoveryMap) {
        const candidate = recoveryMap.get(save.name);
        const reason = candidate?.reason || inferredSaveReason(save.name);
        return `${save.name} (${reasonLabel(reason)}) • ${modifiedTimeLabel(save.modified)}`;
    }

    function renderRecoveryCandidates(candidates) {
        els.recoveryCandidates.innerHTML = '';
        const list = Array.isArray(candidates) ? candidates : [];
        if (!list.length) {
            const li = document.createElement('li');
            li.textContent = 'No recovery candidates available.';
            els.recoveryCandidates.appendChild(li);
            return;
        }
        list.forEach((candidate) => {
            const li = document.createElement('li');
            li.className = 'note-action-item';

            const text = document.createElement('span');
            text.className = 'note-action-text';
            text.textContent = `${candidate.name} | ${candidate.reason} | score ${candidate.score}`;

            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'small-button';
            button.textContent = 'Load';
            button.addEventListener('click', () => {
                loadSaveByName(candidate.name);
            });

            li.appendChild(text);
            li.appendChild(button);
            els.recoveryCandidates.appendChild(li);
        });
    }

    function renderSaveSelector(saves, recovery) {
        latestSaves = Array.isArray(saves) ? saves.slice() : [];
        latestRecovery = recovery || {};
        const recoveryMap = buildRecoveryMap(latestRecovery);
        const prior = els.saveSelect.value;
        const recommended = latestRecovery?.current_recommendation?.name || '';
        const ordered = latestSaves
            .slice()
            .sort((a, b) => (b.modified || 0) - (a.modified || 0));

        els.saveSelect.innerHTML = '';
        if (!ordered.length) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No save states found';
            els.saveSelect.appendChild(option);
            return;
        }

        ordered.forEach((save) => {
            const option = document.createElement('option');
            option.value = save.name;
            option.textContent = optionLabelForSave(save, recoveryMap);
            els.saveSelect.appendChild(option);
        });

        if (prior && ordered.some((save) => save.name === prior)) {
            els.saveSelect.value = prior;
        } else if (recommended && ordered.some((save) => save.name === recommended)) {
            els.saveSelect.value = recommended;
        } else {
            els.saveSelect.selectedIndex = 0;
        }
    }

    function transcriptKey(entry) {
        const meta = entry?.meta ? JSON.stringify(entry.meta) : '';
        return [
            entry?.timestamp || '',
            entry?.direction || '',
            entry?.role || '',
            entry?.channel || '',
            entry?.content || '',
            meta,
        ].join('::');
    }

    function transcriptRole(entry) {
        return entry?.role || (entry?.direction === 'outbound' ? 'user' : entry?.direction === 'inbound' ? 'assistant' : 'system');
    }

    function transcriptRoleLabel(role) {
        if (role === 'assistant_thinking') return 'assistant thinking';
        return role || 'system';
    }

    function createTranscriptCard(entry) {
        const article = document.createElement('article');
        article.className = 'transcript-card';
        article.dataset.role = transcriptRole(entry);

        const meta = document.createElement('div');
        meta.className = 'transcript-meta';

        const chipWrap = document.createElement('div');
        chipWrap.className = 'transcript-chip-row';

        const roleChip = document.createElement('span');
        roleChip.className = 'transcript-chip transcript-role-chip';
        roleChip.dataset.role = transcriptRole(entry);
        roleChip.textContent = transcriptRoleLabel(transcriptRole(entry));

        const channelChip = document.createElement('span');
        channelChip.className = 'transcript-chip transcript-channel-chip';
        channelChip.dataset.direction = entry.direction || 'system';
        channelChip.textContent = entry.channel || 'message';

        const time = document.createElement('span');
        time.textContent = timeLabel(entry.timestamp);

        chipWrap.appendChild(roleChip);
        chipWrap.appendChild(channelChip);

        meta.appendChild(chipWrap);
        meta.appendChild(time);

        const pre = document.createElement('pre');
        pre.textContent = entry.content || '';

        article.appendChild(meta);
        article.appendChild(pre);
        return article;
    }

    function renderTranscript(entries) {
        els.piTranscript.innerHTML = '';
        transcriptKeys.clear();
        const list = Array.isArray(entries) ? entries : [];
        if (!list.length) {
            els.piTranscript.innerHTML = '<p class="empty">No chat messages yet.</p>';
            return;
        }
        list.forEach((entry) => {
            const key = transcriptKey(entry);
            transcriptKeys.add(key);
            els.piTranscript.appendChild(createTranscriptCard(entry));
        });
        els.piTranscript.scrollTop = els.piTranscript.scrollHeight;
    }

    function appendTranscriptEntry(entry) {
        if (!entry || !els.piTranscript) return;
        if (els.piTranscript.querySelector('.empty')) {
            els.piTranscript.innerHTML = '';
        }
        const key = transcriptKey(entry);
        if (transcriptKeys.has(key)) return;
        transcriptKeys.add(key);
        const sticky =
            els.piTranscript.scrollHeight - els.piTranscript.scrollTop - els.piTranscript.clientHeight <
            40;
        while (els.piTranscript.children.length >= TRANSCRIPT_LIMIT) {
            const first = els.piTranscript.firstElementChild;
            if (!first) break;
            els.piTranscript.removeChild(first);
        }
        els.piTranscript.appendChild(createTranscriptCard(entry));
        if (sticky) {
            els.piTranscript.scrollTop = els.piTranscript.scrollHeight;
        }
    }

    function renderWorldStats(worldState, progress, serverRuntime) {
        const map = worldState.map || {};
        const player = worldState.player || {};
        const pos = player.position || {};
        const battle = worldState.battle || {};
        const realtimeLabel = serverRuntime?.realtime_enabled
            ? `${serverRuntime.realtime_fps || 60} FPS`
            : 'paused';
        renderKeyValueCards(els.worldStats, [
            ['Map', map.map_name || 'Unknown'],
            ['Coords', `${pos.x ?? '--'}, ${pos.y ?? '--'}`],
            ['Facing', player.facing || 'unknown'],
            ['Battle', battle.in_battle ? (battle.type || 'active') : 'no'],
            ['Progress', `${progress ?? 0}%`],
            ['Clock', realtimeLabel],
        ]);
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
            const summary =
                event.summary ||
                event.reason ||
                event.text ||
                event.objective?.title ||
                event.tool_name ||
                event.type;
            article.innerHTML = `
                <div class="timeline-head">
                    <span class="timeline-type">${event.type}</span>
                    <span class="timeline-time">${timeLabel(event.timestamp)}</span>
                </div>
                <p>${truncate(summary, 260)}</p>
            `;
            els.timeline.appendChild(article);
        });
    }

    function seedSupervisorControls(supervisor) {
        const config = supervisor.config || {};
        if (controlSeeded) return;
        els.piPromptInput.value = supervisor.default_prompt || '';
        els.piProviderInput.value = config.provider || '';
        els.piModelInput.value = config.model || '';
        els.piThinkingSelect.value = config.thinking || '';
        els.piContinueMessageInput.value = config.continue_message || 'continue';
        els.piAutoContinueInput.checked = Boolean(config.auto_continue ?? true);
        controlSeeded = true;
    }

    function renderSupervisor(supervisor) {
        const config = supervisor.config || {};
        seedSupervisorControls(supervisor);

        const label = supervisor.available ? `${supervisor.status || 'idle'}` : 'Pi unavailable';
        setPiStatus(supervisor.status, label);
        els.piModelChip.textContent = supervisor.model
            ? `Pi: ${supervisor.model}`
            : `Pi: ${supervisor.pi_binary ? 'default model' : 'not installed'}`;
        els.piSessionChip.textContent = supervisor.session_id
            ? `Session: ${supervisor.session_id}`
            : 'Session: none';
        els.piTurnsChip.textContent = `Turns: ${supervisor.turns_completed || 0}`;
        els.piStatusSummary.textContent =
            supervisor.status_reason || supervisor.last_error || 'Pi supervisor not started.';

        renderKeyValueCards(els.piSupervisorStats, [
            ['Status', supervisor.status || 'idle'],
            ['Model', supervisor.model || 'default'],
            ['Provider', supervisor.provider || 'default'],
            ['Thinking', supervisor.thinking || 'default'],
            ['Auto', config.auto_continue ? 'on' : 'off'],
            ['Delay', `${config.continue_delay_seconds ?? 1}s`],
            ['Next', supervisor.next_auto_continue_at ? timeLabel(supervisor.next_auto_continue_at) : 'n/a'],
            ['Continue', config.continue_message || 'continue'],
        ]);

        if (supervisor.last_error) {
            els.piControlStatus.textContent = `Last error: ${supervisor.last_error}`;
        } else if (supervisor.next_auto_continue_at) {
            els.piControlStatus.textContent = `Auto-continue scheduled for ${timeLabel(supervisor.next_auto_continue_at)}.`;
        } else {
            els.piControlStatus.textContent = `Last event: ${timeLabel(supervisor.last_event_at)}`;
        }

        const turnPlanPreview = supervisor.turn_plan_preview?.payload || supervisor.turn_plan_preview;
        els.piTurnPlanPreview.textContent = turnPlanPreview
            ? formatJSON(turnPlanPreview)
            : 'No Pi-authored turn plan captured yet.';

        renderToolList(els.piActiveTools, supervisor.active_tools || [], 'No active Pi tool calls.');
        renderToolList(
            els.piRecentTools,
            (supervisor.recent_tools || []).slice().reverse(),
            'No completed Pi tool calls yet.'
        );
        renderTranscript(supervisor.transcript || []);
        renderList(
            els.piRecentEvents,
            (supervisor.recent_events || []).slice().reverse().map((event) => {
                return `${timeLabel(event.timestamp)} | ${event.type} | ${truncate(event.summary, 140)}`;
            }),
            'No recent Pi events.'
        );
        els.piCurrentThinking.textContent =
            supervisor.current_assistant_thinking ||
            supervisor.last_assistant_thinking ||
            'No reasoning stream yet.';
        els.piCurrentOutput.textContent =
            supervisor.current_assistant_text ||
            supervisor.last_assistant_text ||
            'No assistant output yet.';
        els.piStderr.textContent = (supervisor.stderr_tail || []).join('\n') || 'No stderr output.';
        els.rawSupervisor.textContent = formatJSON(supervisor);
    }

    function renderDashboardState(payload) {
        const visuals = payload.visuals || {};
        const intent = payload.agent_intent || {};
        const world = payload.world_state || {};
        const memory = payload.memory_and_progress || {};
        const objective = intent.objective || {};
        const turnPlan = intent.turn_plan || {};
        const recentAction = intent.recent_action || {};
        const movementGuidance = intent.movement_guidance || {};
        const stateDelta = intent.state_delta || {};
        const screenText = visuals.screen_text || {};
        const recovery = memory.recovery || {};
        const workspace = memory.workspace || {};
        const supervisor = payload.pi_supervisor || {};
        const serverRuntime = payload.server_runtime || {};
        const artifactUrls = payload.artifact_urls || {};

        setStatus(true, 'Connected');
        els.lastUpdate.textContent = timeLabel(payload.generated_at);
        els.uiModeChip.textContent = `UI: ${visuals.ui_mode || 'unknown'}`;
        els.realtimeChip.textContent = serverRuntime.realtime_enabled
            ? `Clock: live @ ${serverRuntime.realtime_fps || 60} FPS`
            : 'Clock: paused';
        els.frameTimestamp.textContent = timeLabel(visuals.frame_timestamp);
        els.screenTextSource.textContent = `OCR: ${screenText.source || 'n/a'}`;

        if (artifactUrls.latest_frame_annotated) {
            els.annotatedFrame.src = withCacheBust(
                artifactUrls.latest_frame_annotated,
                visuals.frame_timestamp
            );
        }
        if (artifactUrls.latest_frame) {
            els.rawFrame.src = withCacheBust(artifactUrls.latest_frame, visuals.frame_timestamp);
        }
        els.screenText.textContent = screenText.text || 'No OCR or dialogue text available.';

        els.objectiveTitle.textContent = objective.title || 'No objective yet';
        els.objectiveProgress.textContent = `${objective.progress_percent ?? memory.progress_percent ?? 0}%`;
        els.objectiveSummary.textContent = objective.summary || 'No objective summary.';
        els.objectivePredicate.textContent = objective.completion_predicate || '';
        els.objectiveRoute.textContent = objective.route_hint || '';
        els.progressFill.style.width = `${memory.progress_percent ?? 0}%`;

        els.turnPlanSummary.textContent = turnPlan.summary || 'No turn plan summary.';
        renderList(els.plannedActions, turnPlan.planned_actions, 'No planned actions set.');
        renderList(els.fallbackActions, turnPlan.fallback_actions, 'No fallback actions set.');
        els.turnPlanNotes.textContent = turnPlan.notes || `Plan updated: ${turnPlan.updated_at || 'never'}`;

        els.recentActionSummary.textContent = recentAction.summary || 'No recent action summary.';
        renderList(els.recentActionNotes, recentAction.notes, 'No recent action notes.');
        renderList(els.stateDeltaSummary, stateDelta.summary, 'No state delta summary.');
        renderList(els.movementGuidance, movementGuidance.notes, 'No movement guidance available.');

        renderWorldStats(world, memory.progress_percent, serverRuntime);
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
        renderRecoveryCandidates(recovery.candidates || []);
        renderSaveSelector(latestSaves, recovery);
        if (memory.stuck) {
            els.stuckSignal.textContent = `${memory.stuck.level}: ${memory.stuck.reason}`;
        } else {
            els.stuckSignal.textContent = 'No stuck signal yet.';
        }
        els.knowledgeSummary.textContent = formatJSON(memory.knowledge_graph_summary || {});
        els.workspaceSummary.textContent = formatJSON(workspace);

        renderSupervisor(supervisor);
        els.rawNavigation.textContent = formatJSON(world.navigation || {});
    }

    async function refreshArtifactPanels(payload) {
        const artifactUrls = payload.artifact_urls || {};
        if (!artifactUrls.latest_observation_json) {
            els.rawObservation.textContent = 'No latest_observation.json artifact available.';
            return;
        }
        try {
            const response = await fetch(
                withCacheBust(artifactUrls.latest_observation_json, payload.generated_at)
            );
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            els.rawObservation.textContent = await response.text();
        } catch (error) {
            els.rawObservation.textContent = `Failed to load raw observation: ${error.message || error}`;
        }
    }

    async function fetchDashboardState() {
        const response = await fetch(api('/dashboard/state'));
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        renderDashboardState(payload);
        await refreshArtifactPanels(payload);
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

    async function fetchSaves() {
        const response = await fetch(api('/saves'));
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        latestSaves = payload.saves || [];
        renderSaveSelector(latestSaves, latestRecovery);
        return payload;
    }

    async function postJson(path, body) {
        const response = await fetch(api(path), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: body ? JSON.stringify(body) : '{}',
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload.detail || `HTTP ${response.status}`);
        }
        return payload;
    }

    async function refreshAll() {
        if (refreshInFlight) {
            return refreshInFlight;
        }
        refreshInFlight = (async () => {
            try {
                await Promise.all([fetchDashboardState(), fetchDashboardHistory(), fetchSaves()]);
            } catch (error) {
                setStatus(false, 'Server unavailable');
                throw error;
            } finally {
                refreshInFlight = null;
            }
        })();
        return refreshInFlight;
    }

    function scheduleRefresh(delay = 350) {
        if (refreshTimer) return;
        refreshTimer = window.setTimeout(() => {
            refreshTimer = null;
            refreshAll().catch(() => {});
        }, delay);
    }

    async function startSupervisor() {
        const body = {
            prompt: els.piPromptInput.value.trim(),
            provider: els.piProviderInput.value.trim() || null,
            model: els.piModelInput.value.trim() || null,
            thinking: els.piThinkingSelect.value || null,
            auto_continue: els.piAutoContinueInput.checked,
            continue_message: els.piContinueMessageInput.value.trim() || 'continue',
        };
        els.piControlStatus.textContent = 'Starting Pi...';
        try {
            await postJson('/supervisor/start', body);
            els.piControlStatus.textContent = 'Pi supervisor started.';
            await refreshAll();
        } catch (error) {
            els.piControlStatus.textContent = String(error.message || error);
        }
    }

    async function continueSupervisor() {
        const body = {
            message: els.piContinueMessageInput.value.trim() || 'continue',
        };
        els.piControlStatus.textContent = 'Continuing Pi for one turn...';
        try {
            await postJson('/supervisor/continue', body);
            els.piControlStatus.textContent = 'Manual continue turn started.';
            await refreshAll();
        } catch (error) {
            els.piControlStatus.textContent = String(error.message || error);
        }
    }

    async function stopSupervisor() {
        els.piControlStatus.textContent = 'Stopping Pi...';
        try {
            await postJson('/supervisor/stop');
            els.piControlStatus.textContent = 'Pi supervisor stopped.';
            await refreshAll();
        } catch (error) {
            els.piControlStatus.textContent = String(error.message || error);
        }
    }

    async function saveNow() {
        const name = els.manualSaveNameInput.value.trim() || defaultSaveName();
        els.manualSaveStatus.textContent = `Saving ${name}...`;
        try {
            const payload = await postJson('/save', { name });
            els.manualSaveStatus.textContent = `Saved ${payload.save?.name || name}.`;
            els.manualSaveNameInput.value = '';
            els.loadSaveStatus.textContent = `Saved ${payload.save?.name || name}.`;
            await refreshAll();
            if (payload.save?.name) {
                els.saveSelect.value = payload.save.name;
            }
        } catch (error) {
            els.manualSaveStatus.textContent = String(error.message || error);
        }
    }

    async function loadSaveByName(name) {
        const trimmed = String(name || '').trim();
        if (!trimmed) {
            els.loadSaveStatus.textContent = 'Choose a save first.';
            return;
        }
        els.loadSaveStatus.textContent = `Loading ${trimmed}...`;
        try {
            const payload = await postJson('/load', { name: trimmed });
            els.loadSaveStatus.textContent = `Loaded ${payload.save?.name || trimmed}.`;
            els.saveSelect.value = payload.save?.name || trimmed;
            await refreshAll();
        } catch (error) {
            els.loadSaveStatus.textContent = String(error.message || error);
        }
    }

    async function loadSelectedSave() {
        await loadSaveByName(els.saveSelect.value);
    }

    async function loadRecommendedSave() {
        const name = latestRecovery?.current_recommendation?.name || '';
        if (!name) {
            els.loadSaveStatus.textContent = 'No recommended recovery save is available.';
            return;
        }
        await loadSaveByName(name);
    }

    function appendStderrLine(text) {
        const current = els.piStderr.textContent.trim();
        const next = current && current !== 'No stderr output.' ? `${current}\n${text}` : text;
        els.piStderr.textContent = next;
    }

    function handleWsEvent(event) {
        if (!event || typeof event !== 'object') return;

        if (event.type === 'connected') {
            setStatus(true, 'Connected');
            scheduleRefresh(100);
            return;
        }

        if (event.type === 'pong') {
            return;
        }

        if (event.type === 'pi_prompt_sent') {
            els.piControlStatus.textContent = event.resume
                ? 'Sent auto-continue prompt to Pi.'
                : 'Sent launch prompt to Pi.';
            return;
        }

        if (event.type === 'pi_transcript') {
            appendTranscriptEntry(event.entry);
            return;
        }

        if (event.type === 'pi_text_delta') {
            if (typeof event.text === 'string' && event.text) {
                els.piCurrentOutput.textContent = event.text;
            } else if (typeof event.delta === 'string' && event.delta) {
                const prior = els.piCurrentOutput.textContent === 'No assistant output yet.' ? '' : els.piCurrentOutput.textContent;
                els.piCurrentOutput.textContent = prior + event.delta;
            }
            return;
        }

        if (event.type === 'pi_thinking_delta') {
            if (typeof event.thinking === 'string' && event.thinking) {
                els.piCurrentThinking.textContent = event.thinking;
            } else if (typeof event.delta === 'string' && event.delta) {
                const prior = els.piCurrentThinking.textContent === 'No reasoning stream yet.' ? '' : els.piCurrentThinking.textContent;
                els.piCurrentThinking.textContent = prior + event.delta;
            }
            return;
        }

        if (event.type === 'pi_stderr' || event.type === 'pi_stdout_parse_error') {
            const text = event.text || event.line;
            if (text) {
                appendStderrLine(text);
            }
            return;
        }

        if (event.type === 'pi_turn_launch' || event.type === 'pi_turn_start') {
            els.piCurrentThinking.textContent = 'Waiting for Pi reasoning...';
            els.piCurrentOutput.textContent = 'Waiting for Pi output...';
            els.piControlStatus.textContent = truncate(event.summary || 'Pi turn started.', 220);
            scheduleRefresh(150);
            return;
        }

        if (event.type === 'pi_auto_continue_scheduled') {
            els.piControlStatus.textContent = event.summary || 'Auto-continue scheduled.';
            scheduleRefresh(150);
            return;
        }

        if (event.type === 'screenshot') {
            scheduleRefresh(150);
            return;
        }

        if (event.type === 'save' || event.type === 'load' || event.type === 'recovery') {
            scheduleRefresh(150);
            return;
        }

        scheduleRefresh(250);
    }

    function scheduleReconnect() {
        if (wsReconnectTimer) return;
        wsReconnectTimer = window.setTimeout(() => {
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
            refreshAll().catch(() => {});
        };

        ws.onmessage = (messageEvent) => {
            let payload = null;
            try {
                payload = JSON.parse(messageEvent.data);
            } catch (_) {
                return;
            }
            handleWsEvent(payload);
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
        setPiStatus('idle', 'Pi idle');
        els.piStartButton.addEventListener('click', startSupervisor);
        els.piContinueButton.addEventListener('click', continueSupervisor);
        els.piStopButton.addEventListener('click', stopSupervisor);
        els.manualSaveButton.addEventListener('click', saveNow);
        els.loadSaveButton.addEventListener('click', loadSelectedSave);
        els.loadRecommendedButton.addEventListener('click', loadRecommendedSave);
        refreshAll().catch(() => {});
        connectWS();
        pollTimer = window.setInterval(() => {
            refreshAll().catch(() => {});
        }, POLL_INTERVAL);
        window.addEventListener('beforeunload', () => {
            if (pollTimer) window.clearInterval(pollTimer);
            if (refreshTimer) window.clearTimeout(refreshTimer);
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
