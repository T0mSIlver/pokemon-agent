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
        statusChip: $('statusChip'),
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
        piGoalInput: $('piGoalInput'),
        piProviderInput: $('piProviderInput'),
        piModelInput: $('piModelInput'),
        piThinkingSelect: $('piThinkingSelect'),
        piAutoContinueInput: $('piAutoContinueInput'),
        piStartButton: $('piStartButton'),
        piContinueButton: $('piContinueButton'),
        piStopButton: $('piStopButton'),
        piControlStatus: $('piControlStatus'),
        piTurnPlanPreview: $('piTurnPlanPreview'),
        piToolFeed: $('piToolFeed'),
        piCurrentThinking: $('piCurrentThinking'),
        piCurrentOutput: $('piCurrentOutput'),
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
        timelineSpark: $('timelineSpark'),
        timelineFilters: $('timelineFilters'),
        timelineCounts: $('timelineCounts'),
        rawObservation: $('rawObservation'),
        rawNavigation: $('rawNavigation'),
        rawSupervisor: $('rawSupervisor'),
        hudFrameHp: $('hudFrameHp'),
        hudFrameHpBar: $('hudFrameHpBar'),
        hudFrameMap: $('hudFrameMap'),
        hudFrameCoord: $('hudFrameCoord'),
        hudFrameFacing: $('hudFrameFacing'),
        hudFrameBadges: $('hudFrameBadges'),
        hudFrameProgress: $('hudFrameProgress'),
        hudFrameProgressBar: $('hudFrameProgressBar'),
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
    let latestTimelineEvents = [];
    let sessionOriginMs = null;
    const transcriptKeys = new Set();
    const autoScrollState = {
        transcript: true,
        thinking: true,
        output: true,
    };
    const timelineFilters = new Set(['all']);
    const EVENT_CATEGORIES = [
        { key: 'all',        label: 'ALL' },
        { key: 'action',     label: 'ACTION' },
        { key: 'decision',   label: 'DECISION' },
        { key: 'battle',     label: 'BATTLE' },
        { key: 'checkpoint', label: 'CHECKPOINT' },
        { key: 'save',       label: 'SAVE' },
        { key: 'objective',  label: 'OBJECTIVE' },
        { key: 'warn',       label: 'WARN' },
        { key: 'error',      label: 'ERROR' },
    ];
    const EVENT_COLOR_VAR = {
        action:     'var(--hud-cyan)',
        decision:   'var(--hud-text)',
        battle:     'var(--hud-bad)',
        checkpoint: 'var(--hud-good)',
        save:       'var(--hud-good)',
        load:       'var(--hud-plasma)',
        recovery:   'var(--hud-plasma)',
        objective:  'var(--hud-hazard)',
        warn:       'var(--hud-warn)',
        error:      'var(--hud-bad)',
        screenshot: 'var(--hud-muted)',
    };

    function api(path) {
        return `${window.location.protocol}//${window.location.host}${path}`;
    }

    function wsUrl() {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${proto}//${window.location.host}/ws`;
    }

    function setStatus(connected, label) {
        els.statusDot.classList.toggle('connected', connected);
        if (els.statusChip) {
            els.statusChip.dataset.status = connected ? 'running' : 'error';
        }
        els.statusText.textContent = (label || '').toUpperCase();
    }

    function setPiStatus(status, label) {
        const normalized = status || 'idle';
        els.piStatusChip.dataset.status = normalized;
        els.piStatusDot.className = 'hud-dot status-dot';
        if (normalized === 'running' || normalized === 'starting') {
            els.piStatusDot.classList.add('running', 'connected');
        } else if (normalized === 'error') {
            els.piStatusDot.classList.add('error');
        } else if (normalized === 'stopping') {
            els.piStatusDot.classList.add('warning');
        } else {
            els.piStatusDot.classList.add('idle');
        }
        els.piStatusText.textContent = (label || '').toUpperCase();
    }

    function formatJSON(value) {
        return JSON.stringify(value ?? {}, null, 2);
    }

    function currentFullscreenElement() {
        return document.fullscreenElement || document.webkitFullscreenElement || null;
    }

    function frameViewports() {
        return Array.from(document.querySelectorAll('.hud-frame-view'));
    }

    function isElementFullscreen(element) {
        return Boolean(element) && currentFullscreenElement() === element;
    }

    function frameViewportImage(viewport) {
        if (!viewport) return null;
        const imageId = viewport.dataset.frameImage;
        if (imageId) {
            return document.getElementById(imageId);
        }
        return viewport.querySelector('img');
    }

    function frameViewportLabel(viewport) {
        return viewport?.dataset.frameLabel || 'emulator';
    }

    function syncFrameFullscreenState() {
        const fullscreenElement = currentFullscreenElement();
        frameViewports().forEach((viewport) => {
            const isFullscreen = fullscreenElement === viewport;
            const label = frameViewportLabel(viewport);
            viewport.dataset.fullscreen = isFullscreen ? 'true' : 'false';
            viewport.setAttribute(
                'aria-label',
                isFullscreen
                    ? `Exit fullscreen ${label} view`
                    : `Toggle fullscreen ${label} view`
            );
            viewport.title = isFullscreen
                ? 'Click to exit fullscreen'
                : 'Click to toggle fullscreen';
        });
    }

    async function requestElementFullscreen(element) {
        if (!element) return;
        if (typeof element.requestFullscreen === 'function') {
            await element.requestFullscreen();
            return;
        }
        if (typeof element.webkitRequestFullscreen === 'function') {
            element.webkitRequestFullscreen();
        }
    }

    async function exitFullscreen() {
        if (typeof document.exitFullscreen === 'function') {
            await document.exitFullscreen();
            return;
        }
        if (typeof document.webkitExitFullscreen === 'function') {
            document.webkitExitFullscreen();
        }
    }

    async function toggleFrameFullscreen(viewport) {
        const target = viewport || null;
        const frameImage = frameViewportImage(target);
        if (!target || !frameImage?.src) return;
        try {
            if (isElementFullscreen(target)) {
                await exitFullscreen();
            } else {
                await requestElementFullscreen(target);
            }
        } catch (_) {
            // Ignore rejected fullscreen requests; the browser will enforce gesture rules.
        } finally {
            syncFrameFullscreenState();
        }
    }

    function onFrameViewportKeydown(event) {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        toggleFrameFullscreen(event.currentTarget);
    }

    function onFrameViewportClick(event) {
        toggleFrameFullscreen(event.currentTarget);
    }

    function withCacheBust(url, token) {
        if (!url) return '';
        const suffix = token ? encodeURIComponent(token) : String(Date.now());
        return `${url}${url.includes('?') ? '&' : '?'}t=${suffix}`;
    }

    function formatCompactNumber(value) {
        if (typeof value !== 'number' || !Number.isFinite(value)) return 'n/a';
        if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
        if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
        return String(value);
    }

    function parseCompactTokenCount(value) {
        if (typeof value === 'number' && Number.isFinite(value)) return value;
        if (typeof value !== 'string') return NaN;
        const match = value.trim().match(/^([\d.]+)\s*([KMB])?$/i);
        if (!match) return NaN;
        const amount = Number(match[1]);
        if (!Number.isFinite(amount)) return NaN;
        const suffix = (match[2] || '').toUpperCase();
        if (suffix === 'B') return amount * 1_000_000_000;
        if (suffix === 'M') return amount * 1_000_000;
        if (suffix === 'K') return amount * 1_000;
        return amount;
    }

    function formatContextUsage(usage, limits) {
        if (!usage || typeof usage !== 'object') return 'n/a';
        const total = usage.totalTokens;
        if (typeof total !== 'number') return 'n/a';
        const limitLabel =
            typeof limits?.context_window === 'string' && limits.context_window.trim()
                ? limits.context_window.trim()
                : '';
        const limitTokens = parseCompactTokenCount(
            limits?.context_window_tokens ?? limits?.context_window
        );
        if (Number.isFinite(limitTokens) && limitTokens > 0) {
            const pct = (total / limitTokens) * 100;
            return `${formatCompactNumber(total)} / ${limitLabel || formatCompactNumber(limitTokens)} (${pct.toFixed(0)}%)`;
        }
        return formatCompactNumber(total);
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

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function parseTs(value) {
        if (!value) return NaN;
        if (typeof value === 'number') return value;
        const t = new Date(value).getTime();
        return Number.isNaN(t) ? NaN : t;
    }

    function isNearBottom(node, threshold = 16) {
        if (!node) return false;
        return node.scrollHeight - node.scrollTop - node.clientHeight <= threshold;
    }

    function shouldAutoScroll(name, node) {
        if (!node) return false;
        return autoScrollState[name] || isNearBottom(node);
    }

    function syncAutoScrollState(name, node) {
        if (!node) return;
        autoScrollState[name] = isNearBottom(node);
        node.dataset.autoscroll = autoScrollState[name] ? 'true' : 'false';
    }

    function initAutoScroll(name, node) {
        if (!node) return;
        syncAutoScrollState(name, node);
        node.addEventListener(
            'scroll',
            () => {
                syncAutoScrollState(name, node);
            },
            { passive: true }
        );
    }

    function scrollNodeToBottom(node, name, force = false) {
        if (!node) return;
        if (!force && !shouldAutoScroll(name, node)) return;
        node.scrollTop = node.scrollHeight;
        if (name) {
            syncAutoScrollState(name, node);
        }
    }

    function relTime(ts, originMs) {
        const t = parseTs(ts);
        if (Number.isNaN(t) || !originMs) return '--:--';
        const seconds = Math.max(0, Math.round((t - originMs) / 1000));
        const hh = Math.floor(seconds / 3600);
        const mm = Math.floor((seconds % 3600) / 60);
        const ss = seconds % 60;
        const pad = (n) => String(n).padStart(2, '0');
        if (hh > 0) return `T+${hh}:${pad(mm)}:${pad(ss)}`;
        return `T+${pad(mm)}:${pad(ss)}`;
    }

    function deltaTime(currTs, prevTs) {
        const c = parseTs(currTs);
        const p = parseTs(prevTs);
        if (Number.isNaN(c) || Number.isNaN(p)) return '';
        const diffMs = c - p;
        if (diffMs < 0) return '';
        if (diffMs < 1000) return `Δ ${diffMs}ms`;
        if (diffMs < 60000) return `Δ ${(diffMs / 1000).toFixed(1)}s`;
        const minutes = Math.floor(diffMs / 60000);
        const seconds = Math.round((diffMs % 60000) / 1000);
        return `Δ ${minutes}m${seconds}s`;
    }

    function eventCategory(type) {
        const t = String(type || '').toLowerCase();
        if (!t) return 'default';
        if (t.includes('error') || t.includes('fail') || t.includes('crash')) return 'error';
        if (t.includes('warn') || t.includes('stuck')) return 'warn';
        if (t.includes('objective') || t.includes('goal') || t.includes('progress')) return 'objective';
        if (t.includes('checkpoint')) return 'checkpoint';
        if (t.includes('save')) return 'save';
        if (t.includes('load') || t.includes('recovery') || t.includes('restore')) return 'recovery';
        if (t.includes('battle') || t.includes('combat') || t.includes('faint')) return 'battle';
        if (t.includes('decision') || t.includes('plan') || t.includes('intent')) return 'decision';
        if (t.includes('screenshot') || t.includes('frame')) return 'screenshot';
        if (t.includes('action') || t.includes('tool') || t.includes('move') || t.includes('navigate')) return 'action';
        return 'action';
    }

    function eventIcon(category) {
        switch (category) {
            case 'action':     return '◉';
            case 'decision':   return '◈';
            case 'battle':     return '⬢';
            case 'checkpoint': return '★';
            case 'save':       return '★';
            case 'load':       return '↺';
            case 'recovery':   return '↺';
            case 'objective':  return '⚑';
            case 'warn':       return '⚠';
            case 'error':      return '✕';
            case 'screenshot': return '▸';
            default:           return '⟡';
        }
    }

    function kvPill(label, value) {
        if (value === null || value === undefined || value === '') return '';
        return `<span class="hud-kv-pill">${escapeHtml(label)} <strong>${escapeHtml(value)}</strong></span>`;
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
        if (!node) return;
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
        if (!node) return;
        node.innerHTML = '';
        pairs.forEach(([label, value, tone]) => {
            const card = document.createElement('div');
            card.className = 'hud-stat';
            if (tone) card.dataset.type = tone;
            const lbl = document.createElement('span');
            lbl.className = 'hud-stat-label';
            lbl.textContent = label;
            const val = document.createElement('strong');
            val.className = 'hud-stat-value';
            val.textContent = value || 'n/a';
            card.appendChild(lbl);
            card.appendChild(val);
            node.appendChild(card);
        });
    }

    function toolKey(item) {
        return (
            item?.tool_call_id ||
            [
                item?.tool_name || '',
                item?.summary || '',
                item?.started_at || '',
                item?.finished_at || '',
                item?.status || '',
            ].join('::')
        );
    }

    function toolStatusLabel(status) {
        const normalized = String(status || 'completed').toLowerCase();
        if (normalized === 'running') return 'LIVE';
        if (normalized === 'completed') return 'DONE';
        return normalized.toUpperCase();
    }

    function toolTimeLabel(item) {
        const stamp = item?.finished_at || item?.started_at || '';
        const abs = timeLabel(stamp);
        const rel = sessionOriginMs ? relTime(stamp, sessionOriginMs) : '';
        return rel || abs;
    }

    function toolText(value, fallback) {
        const text = String(value ?? '').trim();
        return text || fallback;
    }

    function createToolCard(item) {
        const card = document.createElement('details');
        card.className = 'hud-tool-card';

        const summary = document.createElement('summary');

        const summaryBody = document.createElement('div');
        summaryBody.className = 'hud-tool-summary';

        const title = document.createElement('strong');
        title.className = 'hud-tool-title';
        title.dataset.slot = 'title';

        const preview = document.createElement('p');
        preview.className = 'hud-tool-preview';
        preview.dataset.slot = 'preview';

        summaryBody.appendChild(title);
        summaryBody.appendChild(preview);

        const meta = document.createElement('div');
        meta.className = 'hud-tool-meta';

        const status = document.createElement('span');
        status.className = 'hud-chip hud-chip--tiny';
        status.dataset.slot = 'status';

        const time = document.createElement('span');
        time.className = 'hud-chat-time';
        time.dataset.slot = 'time';

        meta.appendChild(status);
        meta.appendChild(time);

        summary.appendChild(summaryBody);
        summary.appendChild(meta);

        const body = document.createElement('div');
        body.className = 'hud-tool-body';

        const argsBlock = document.createElement('div');
        argsBlock.className = 'hud-tool-block';
        const argsLabel = document.createElement('span');
        argsLabel.className = 'hud-tool-block-label';
        argsLabel.textContent = 'Arguments';
        const argsPre = document.createElement('pre');
        argsPre.dataset.slot = 'args';
        argsBlock.appendChild(argsLabel);
        argsBlock.appendChild(argsPre);

        const resultBlock = document.createElement('div');
        resultBlock.className = 'hud-tool-block';
        const resultLabel = document.createElement('span');
        resultLabel.className = 'hud-tool-block-label';
        resultLabel.textContent = 'Output';
        const resultPre = document.createElement('pre');
        resultPre.dataset.slot = 'result';
        resultBlock.appendChild(resultLabel);
        resultBlock.appendChild(resultPre);

        body.appendChild(argsBlock);
        body.appendChild(resultBlock);

        card.appendChild(summary);
        card.appendChild(body);
        updateToolCard(card, item);
        return card;
    }

    function updateToolCard(card, item) {
        if (!card) return;
        const key = toolKey(item);
        const status = String(item?.status || 'completed').toLowerCase();
        const title = card.querySelector('[data-slot="title"]');
        const preview = card.querySelector('[data-slot="preview"]');
        const statusChip = card.querySelector('[data-slot="status"]');
        const time = card.querySelector('[data-slot="time"]');
        const args = card.querySelector('[data-slot="args"]');
        const result = card.querySelector('[data-slot="result"]');

        card.dataset.toolKey = key;
        card.dataset.status = status;

        if (title) {
            title.textContent = truncate(item.summary || item.tool_name || 'tool', 120);
        }
        if (preview) {
            preview.textContent = truncate(
                item.result_preview || item.args_preview || 'No preview available.',
                180
            );
        }
        if (statusChip) {
            statusChip.textContent = toolStatusLabel(status);
        }
        if (time) {
            time.textContent = toolTimeLabel(item);
            time.title = item?.finished_at
                ? `Finished ${timeLabel(item.finished_at)}`
                : `Started ${timeLabel(item?.started_at)}`;
        }
        if (args) {
            args.textContent = toolText(item?.args, 'No arguments captured.');
        }
        if (result) {
            result.textContent = toolText(
                item?.result,
                status === 'running' ? 'Waiting for tool output…' : 'No output captured.'
            );
        }
    }

    function renderToolList(node, items, fallback) {
        if (!node) return;
        const list = Array.isArray(items) ? items : [];
        if (!list.length) {
            const empty = document.createElement('p');
            empty.className = 'hud-empty';
            empty.textContent = fallback;
            node.replaceChildren(empty);
            return;
        }

        const empty = node.querySelector('.hud-empty');
        if (empty) empty.remove();

        const existingCards = Array.from(node.querySelectorAll('.hud-tool-card'));
        const existingByKey = new Map(
            existingCards.map((card) => [card.dataset.toolKey || '', card])
        );
        const nextKeys = new Set();

        list.forEach((item, index) => {
            const key = toolKey(item);
            nextKeys.add(key);
            let card = existingByKey.get(key);
            if (!card) {
                card = createToolCard(item);
            } else {
                updateToolCard(card, item);
            }
            const currentChild = node.children[index];
            if (currentChild !== card) {
                node.insertBefore(card, currentChild || null);
            }
        });

        Array.from(node.querySelectorAll('.hud-tool-card')).forEach((card) => {
            if (!nextKeys.has(card.dataset.toolKey || '')) {
                card.remove();
            }
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
            li.className = 'hud-recovery-item';

            const text = document.createElement('span');
            text.className = 'hud-recovery-text';
            text.textContent = `${candidate.name} · ${candidate.reason} · score ${candidate.score}`;

            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'hud-btn hud-btn--small hud-btn--ghost';
            button.textContent = 'LOAD';
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
        article.className = 'hud-chat-card';

        const meta = document.createElement('div');
        meta.className = 'hud-chat-meta';

        const chipWrap = document.createElement('div');
        chipWrap.className = 'hud-chat-chips';

        const roleChip = document.createElement('span');
        roleChip.className = 'hud-chat-chip';
        roleChip.dataset.slot = 'role';

        const channelChip = document.createElement('span');
        channelChip.className = 'hud-chat-chip';
        channelChip.dataset.slot = 'channel';

        const time = document.createElement('span');
        time.className = 'hud-chat-time';
        time.dataset.slot = 'time';

        chipWrap.appendChild(roleChip);
        chipWrap.appendChild(channelChip);

        meta.appendChild(chipWrap);
        meta.appendChild(time);

        const pre = document.createElement('pre');
        pre.dataset.slot = 'content';

        article.appendChild(meta);
        article.appendChild(pre);
        updateTranscriptCard(article, entry);
        return article;
    }

    function updateTranscriptCard(article, entry) {
        if (!article) return;
        const role = transcriptRole(entry);
        const key = transcriptKey(entry);
        const roleChip = article.querySelector('[data-slot="role"]');
        const channelChip = article.querySelector('[data-slot="channel"]');
        const time = article.querySelector('[data-slot="time"]');
        const pre = article.querySelector('[data-slot="content"]');

        article.dataset.role = role;
        article.dataset.transcriptKey = key;

        if (roleChip) {
            roleChip.dataset.role = role;
            roleChip.textContent = transcriptRoleLabel(role).toUpperCase();
        }
        if (channelChip) {
            channelChip.dataset.channel = entry.channel || 'message';
            channelChip.dataset.direction = entry.direction || 'system';
            channelChip.textContent = (entry.channel || 'message').toUpperCase();
        }
        if (time) {
            const abs = timeLabel(entry.timestamp);
            const rel = sessionOriginMs ? relTime(entry.timestamp, sessionOriginMs) : '';
            time.textContent = rel || abs;
            time.title = abs;
        }
        if (pre) {
            pre.textContent = entry.content || '';
        }
    }

    function scrollTranscriptToBottom(force = false) {
        scrollNodeToBottom(els.piTranscript, 'transcript', force);
    }

    function renderTranscript(entries) {
        if (!els.piTranscript) return;
        const list = (Array.isArray(entries) ? entries : []).slice(-TRANSCRIPT_LIMIT);
        const nextKeys = new Set();
        const follow = shouldAutoScroll('transcript', els.piTranscript);

        transcriptKeys.clear();
        if (!list.length) {
            const empty = document.createElement('p');
            empty.className = 'hud-empty';
            empty.textContent = '— awaiting comms —';
            els.piTranscript.replaceChildren(empty);
            scrollTranscriptToBottom(follow);
            return;
        }

        const empty = els.piTranscript.querySelector('.hud-empty');
        if (empty) empty.remove();

        const existingCards = Array.from(els.piTranscript.querySelectorAll('.hud-chat-card'));
        const existingByKey = new Map(
            existingCards.map((card) => [card.dataset.transcriptKey || '', card])
        );

        list.forEach((entry, index) => {
            const key = transcriptKey(entry);
            nextKeys.add(key);
            transcriptKeys.add(key);
            let card = existingByKey.get(key);
            if (!card) {
                card = createTranscriptCard(entry);
            } else {
                updateTranscriptCard(card, entry);
            }
            const currentChild = els.piTranscript.children[index];
            if (currentChild !== card) {
                els.piTranscript.insertBefore(card, currentChild || null);
            }
        });

        Array.from(els.piTranscript.querySelectorAll('.hud-chat-card')).forEach((card) => {
            if (!nextKeys.has(card.dataset.transcriptKey || '')) {
                card.remove();
            }
        });

        scrollTranscriptToBottom(follow);
    }

    function appendTranscriptEntry(entry) {
        if (!entry || !els.piTranscript) return;
        const follow = shouldAutoScroll('transcript', els.piTranscript);
        const empty = els.piTranscript.querySelector('.hud-empty');
        if (empty) empty.remove();
        const key = transcriptKey(entry);
        if (transcriptKeys.has(key)) return;
        transcriptKeys.add(key);
        while (els.piTranscript.children.length >= TRANSCRIPT_LIMIT) {
            const first = els.piTranscript.firstElementChild;
            if (!first) break;
            const firstKey = first.dataset.transcriptKey || '';
            if (firstKey) {
                transcriptKeys.delete(firstKey);
            }
            els.piTranscript.removeChild(first);
        }
        els.piTranscript.appendChild(createTranscriptCard(entry));
        scrollTranscriptToBottom(follow);
    }

    function renderWorldStats(worldState, progress, serverRuntime) {
        const map = worldState.map || {};
        const player = worldState.player || {};
        const pos = player.position || {};
        const battle = worldState.battle || {};
        const realtimeLabel = serverRuntime?.realtime_enabled
            ? `${serverRuntime.realtime_fps || 60}/${serverRuntime.live_artifact_fps || 0} FPS`
            : 'paused';
        renderKeyValueCards(els.worldStats, [
            ['MAP', map.map_name || 'Unknown'],
            ['COORDS', `${pos.x ?? '--'}, ${pos.y ?? '--'}`],
            ['FACING', player.facing || 'unknown'],
            ['BATTLE', battle.in_battle ? (battle.type || 'active') : 'no'],
            ['PROGRESS', `${progress ?? 0}%`, 'progress'],
            ['CLOCK', realtimeLabel],
        ]);
    }

    function hpTone(ratio) {
        if (ratio >= 0.5) return 'good';
        if (ratio >= 0.2) return 'mid';
        return 'low';
    }

    function renderFrameHud(worldState, memory) {
        const party = Array.isArray(worldState?.party) ? worldState.party : [];
        const totals = party.reduce(
            (acc, mon) => {
                const hp = Number(mon.hp) || 0;
                const max = Number(mon.max_hp) || 0;
                acc.hp += hp;
                acc.max += max;
                return acc;
            },
            { hp: 0, max: 0 }
        );
        const hpRatio = totals.max > 0 ? totals.hp / totals.max : 0;
        if (els.hudFrameHp) {
            els.hudFrameHp.textContent = totals.max > 0
                ? `${totals.hp} / ${totals.max}`
                : '—';
        }
        if (els.hudFrameHpBar) {
            els.hudFrameHpBar.style.width = `${Math.min(100, Math.round(hpRatio * 100))}%`;
            els.hudFrameHpBar.dataset.tone = hpTone(hpRatio);
        }
        const map = worldState?.map || {};
        const player = worldState?.player || {};
        const pos = player.position || {};
        if (els.hudFrameMap) {
            els.hudFrameMap.textContent = map.map_name || map.id || '—';
        }
        if (els.hudFrameCoord) {
            els.hudFrameCoord.textContent =
                pos.x !== undefined && pos.y !== undefined
                    ? `${pos.x}, ${pos.y}`
                    : '—';
        }
        if (els.hudFrameFacing) {
            els.hudFrameFacing.textContent = (player.facing || '—').toString().toUpperCase();
        }
        if (els.hudFrameBadges) {
            const badges =
                memory?.badges ??
                memory?.badge_count ??
                (Array.isArray(player?.badges) ? player.badges.length : null);
            els.hudFrameBadges.textContent = badges !== null && badges !== undefined ? String(badges) : '—';
        }
        const progressPct = memory?.progress_percent ?? 0;
        if (els.hudFrameProgress) {
            els.hudFrameProgress.textContent = `${progressPct}%`;
        }
        if (els.hudFrameProgressBar) {
            els.hudFrameProgressBar.style.width = `${Math.min(100, Math.max(0, Number(progressPct) || 0))}%`;
        }
    }

    function renderParty(party) {
        els.partySnapshot.innerHTML = '';
        const list = Array.isArray(party) ? party : [];
        if (!list.length) {
            els.partySnapshot.innerHTML = '<p class="hud-empty">— no party data —</p>';
            return;
        }
        list.forEach((mon) => {
            const card = document.createElement('article');
            card.className = 'hud-party-card';
            const name = mon.nickname || mon.species || 'Unknown';
            const species = mon.species && mon.nickname ? mon.species : '';
            const level = mon.level ? `LV ${mon.level}` : 'LV —';
            const hp = Number(mon.hp) || 0;
            const maxHp = Number(mon.max_hp) || 0;
            const ratio = maxHp > 0 ? hp / maxHp : 0;
            const tone = hpTone(ratio);
            const status = (mon.status || 'OK').toString();
            const statusTone = /^(ok|none|healthy)$/i.test(status)
                ? ''
                : /(par|slp|brn|psn|frz)/i.test(status)
                ? 'warn'
                : 'bad';
            const moves = Array.isArray(mon.moves)
                ? mon.moves.map((m) => (typeof m === 'string' ? m : m.name || '—'))
                : [];

            const head = document.createElement('div');
            head.className = 'hud-party-head';
            const nameEl = document.createElement('span');
            nameEl.className = 'hud-party-name';
            nameEl.textContent = name;
            const lvEl = document.createElement('span');
            lvEl.className = 'hud-party-lv';
            lvEl.textContent = level;
            head.appendChild(nameEl);
            head.appendChild(lvEl);
            card.appendChild(head);

            if (species || mon.type) {
                const sub = document.createElement('div');
                sub.className = 'hud-party-sub';
                if (species) {
                    const s = document.createElement('span');
                    s.innerHTML = `<strong>${escapeHtml(species)}</strong>`;
                    sub.appendChild(s);
                }
                if (mon.type) {
                    const t = document.createElement('span');
                    t.textContent = mon.type;
                    sub.appendChild(t);
                }
                card.appendChild(sub);
            }

            const hpRow = document.createElement('div');
            hpRow.className = 'hud-party-hp-row';
            const hpLbl = document.createElement('span');
            hpLbl.className = 'hud-party-hp-label';
            hpLbl.textContent = 'HP';
            const hpBar = document.createElement('div');
            hpBar.className = 'hud-party-hp-bar';
            const hpFill = document.createElement('div');
            hpFill.className = 'hud-party-hp-fill';
            hpFill.dataset.tone = tone;
            hpFill.style.width = `${Math.min(100, Math.round(ratio * 100))}%`;
            hpBar.appendChild(hpFill);
            const hpVal = document.createElement('span');
            hpVal.className = 'hud-party-hp-value';
            hpVal.textContent = maxHp > 0 ? `${hp} / ${maxHp}` : '— / —';
            hpRow.appendChild(hpLbl);
            hpRow.appendChild(hpBar);
            hpRow.appendChild(hpVal);
            card.appendChild(hpRow);

            const statusEl = document.createElement('span');
            statusEl.className = 'hud-party-status';
            if (statusTone) statusEl.dataset.tone = statusTone;
            statusEl.textContent = `STATUS · ${status.toUpperCase()}`;
            card.appendChild(statusEl);

            if (moves.length) {
                const movesEl = document.createElement('div');
                movesEl.className = 'hud-party-moves';
                moves.forEach((move) => {
                    const m = document.createElement('span');
                    m.className = 'hud-party-move';
                    m.textContent = move;
                    movesEl.appendChild(m);
                });
                card.appendChild(movesEl);
            }

            els.partySnapshot.appendChild(card);
        });
    }

    function renderTimelineFilters(typeCounts) {
        if (!els.timelineFilters) return;
        els.timelineFilters.innerHTML = '';
        EVENT_CATEGORIES.forEach(({ key, label }) => {
            const count = key === 'all' ? latestTimelineEvents.length : (typeCounts[key] || 0);
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'hud-filter-chip';
            const active = timelineFilters.has(key) || (timelineFilters.size === 0 && key === 'all');
            chip.dataset.active = active ? 'true' : 'false';
            chip.dataset.filter = key;
            chip.innerHTML = `${escapeHtml(label)}<span class="hud-filter-count">${count}</span>`;
            chip.addEventListener('click', () => toggleTimelineFilter(key));
            els.timelineFilters.appendChild(chip);
        });
    }

    function toggleTimelineFilter(key) {
        if (key === 'all') {
            timelineFilters.clear();
            timelineFilters.add('all');
        } else {
            timelineFilters.delete('all');
            if (timelineFilters.has(key)) {
                timelineFilters.delete(key);
            } else {
                timelineFilters.add(key);
            }
            if (timelineFilters.size === 0) {
                timelineFilters.add('all');
            }
        }
        renderTimeline(latestTimelineEvents);
    }

    function renderTimelineSparkline(events) {
        if (!els.timelineSpark) return;
        const svg = els.timelineSpark;
        svg.innerHTML = '';
        const BUCKETS = 30;
        const WIDTH = 300;
        const HEIGHT = 36;
        const bucketW = WIDTH / BUCKETS;
        if (!events.length) return;
        const stamps = events
            .map((e) => parseTs(e.timestamp))
            .filter((t) => !Number.isNaN(t));
        if (!stamps.length) return;
        const minT = Math.min(...stamps);
        const maxT = Math.max(...stamps);
        const range = Math.max(1, maxT - minT);
        const buckets = Array.from({ length: BUCKETS }, () => ({}));
        events.forEach((event) => {
            const t = parseTs(event.timestamp);
            if (Number.isNaN(t)) return;
            const idx = Math.min(BUCKETS - 1, Math.floor(((t - minT) / range) * BUCKETS));
            const cat = eventCategory(event.type);
            buckets[idx][cat] = (buckets[idx][cat] || 0) + 1;
        });
        let maxStack = 1;
        buckets.forEach((b) => {
            const sum = Object.values(b).reduce((a, c) => a + c, 0);
            if (sum > maxStack) maxStack = sum;
        });
        buckets.forEach((bucket, idx) => {
            const entries = Object.entries(bucket);
            if (!entries.length) return;
            let yOffset = HEIGHT;
            entries.forEach(([cat, count]) => {
                const h = (count / maxStack) * (HEIGHT - 2);
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', String(idx * bucketW + 0.5));
                rect.setAttribute('y', String(yOffset - h));
                rect.setAttribute('width', String(Math.max(1, bucketW - 1)));
                rect.setAttribute('height', String(h));
                rect.setAttribute('fill', EVENT_COLOR_VAR[cat] || 'var(--hud-cyan)');
                rect.setAttribute('opacity', '0.72');
                svg.appendChild(rect);
                yOffset -= h;
            });
        });
        const axis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        axis.setAttribute('x1', '0');
        axis.setAttribute('x2', String(WIDTH));
        axis.setAttribute('y1', String(HEIGHT - 0.5));
        axis.setAttribute('y2', String(HEIGHT - 0.5));
        axis.setAttribute('stroke', 'var(--hud-line)');
        axis.setAttribute('stroke-width', '1');
        svg.appendChild(axis);
    }

    function eventSummaryText(event) {
        return (
            event.summary ||
            event.reason ||
            event.text ||
            event.objective?.title ||
            event.tool_name ||
            event.type ||
            '(no detail)'
        );
    }

    function renderTimelineCounts(total) {
        if (!els.timelineCounts) return;
        els.timelineCounts.innerHTML = '';
        const chip = document.createElement('span');
        chip.className = 'hud-chip hud-chip--tiny';
        chip.textContent = `${total} EVENTS`;
        els.timelineCounts.appendChild(chip);
    }

    function timelineEventKey(event) {
        return JSON.stringify({
            timestamp: event?.timestamp || '',
            type: event?.type || '',
            summary: event?.summary || '',
            reason: event?.reason || '',
            text: event?.text || '',
            tool_name: event?.tool_name || '',
            action: event?.action || '',
            map_name: event?.map_name || '',
            coords: event?.coords || '',
            status: event?.status || '',
            duration_ms: event?.duration_ms ?? '',
            outcome: event?.outcome || '',
        });
    }

    function createTimelineEvent() {
        const article = document.createElement('article');
        article.className = 'hud-event';

        const gutter = document.createElement('div');
        gutter.className = 'hud-event-gutter';
        gutter.dataset.slot = 'gutter';

        const icon = document.createElement('div');
        icon.className = 'hud-event-icon';
        icon.dataset.slot = 'icon';

        const type = document.createElement('div');
        type.className = 'hud-event-type';
        type.dataset.slot = 'type';

        const summary = document.createElement('div');
        summary.className = 'hud-event-summary';
        summary.dataset.slot = 'summary';

        const meta = document.createElement('div');
        meta.className = 'hud-event-meta';
        meta.dataset.slot = 'meta';

        article.appendChild(gutter);
        article.appendChild(icon);
        article.appendChild(type);
        article.appendChild(summary);
        article.appendChild(meta);
        return article;
    }

    function updateTimelineEvent(article, event, previousEvent) {
        const category = eventCategory(event.type);
        const gutter = article.querySelector('[data-slot="gutter"]');
        const icon = article.querySelector('[data-slot="icon"]');
        const type = article.querySelector('[data-slot="type"]');
        const summary = article.querySelector('[data-slot="summary"]');
        const meta = article.querySelector('[data-slot="meta"]');
        const summaryText = eventSummaryText(event);
        const pills = [];

        article.dataset.eventKey = timelineEventKey(event);
        article.dataset.type = category;
        delete article.dataset.clusterContinue;

        if (previousEvent) {
            const diff = Math.abs(parseTs(event.timestamp) - parseTs(previousEvent.timestamp));
            if (!Number.isNaN(diff) && diff < 1000) {
                article.dataset.clusterContinue = 'true';
            }
        }

        if (gutter) {
            const rel = relTime(event.timestamp, sessionOriginMs);
            const delta = previousEvent ? deltaTime(event.timestamp, previousEvent.timestamp) : '';
            gutter.innerHTML = `
                <span>${escapeHtml(rel)}</span>
                ${delta ? `<span class="hud-event-delta">${escapeHtml(delta)}</span>` : ''}
            `;
        }
        if (icon) {
            icon.textContent = eventIcon(category);
        }
        if (type) {
            type.textContent = (event.type || category).toString().toUpperCase();
        }
        if (event.tool_name) pills.push(kvPill('tool', event.tool_name));
        if (event.action) pills.push(kvPill('act', event.action));
        if (event.map_name) pills.push(kvPill('map', event.map_name));
        if (event.coords) pills.push(kvPill('at', event.coords));
        if (typeof event.duration_ms === 'number') pills.push(kvPill('ms', event.duration_ms));
        if (event.outcome) pills.push(kvPill('→', event.outcome));
        if (summary) {
            summary.innerHTML = `
                <span>${escapeHtml(truncate(summaryText, 260))}</span>
                ${pills.length ? `<span class="hud-kv">${pills.join('')}</span>` : ''}
            `;
        }
        if (meta) {
            meta.innerHTML = '';
            if (event.reason && event.reason !== summaryText) {
                const reason = document.createElement('span');
                reason.textContent = truncate(event.reason, 40);
                meta.appendChild(reason);
            }
            if (event.status) {
                const status = document.createElement('span');
                status.textContent = event.status;
                meta.appendChild(status);
            }
        }
    }

    function renderTimeline(events) {
        const recent = Array.isArray(events) ? events : [];
        latestTimelineEvents = recent;
        const ordered = recent.slice().sort((a, b) => {
            const ta = parseTs(a.timestamp);
            const tb = parseTs(b.timestamp);
            if (Number.isNaN(ta) || Number.isNaN(tb)) return 0;
            return ta - tb;
        });

        const stamps = ordered
            .map((e) => parseTs(e.timestamp))
            .filter((t) => !Number.isNaN(t));
        const originFromEvents = stamps.length ? Math.min(...stamps) : null;
        if (!sessionOriginMs || (originFromEvents && originFromEvents < sessionOriginMs)) {
            sessionOriginMs = originFromEvents;
        }

        const typeCounts = {};
        ordered.forEach((e) => {
            const cat = eventCategory(e.type);
            typeCounts[cat] = (typeCounts[cat] || 0) + 1;
        });

        renderTimelineCounts(ordered.length);
        renderTimelineFilters(typeCounts);
        renderTimelineSparkline(ordered);

        if (!ordered.length) {
            const empty = document.createElement('p');
            empty.className = 'hud-empty';
            empty.textContent = '[ NO TRAFFIC ]';
            els.timeline.replaceChildren(empty);
            return;
        }

        const showAll = timelineFilters.has('all');
        const visible = ordered.filter((event) => {
            if (showAll) return true;
            const cat = eventCategory(event.type);
            return timelineFilters.has(cat);
        });

        if (!visible.length) {
            const empty = document.createElement('p');
            empty.className = 'hud-empty';
            empty.textContent = '[ NO MATCHING EVENTS ]';
            els.timeline.replaceChildren(empty);
            return;
        }

        const reversed = visible.slice().reverse();
        const empty = els.timeline.querySelector('.hud-empty');
        if (empty) empty.remove();

        const existingCards = Array.from(els.timeline.querySelectorAll('.hud-event'));
        const existingByKey = new Map(
            existingCards.map((card) => [card.dataset.eventKey || '', card])
        );
        const nextKeys = new Set();

        reversed.forEach((event, idx) => {
            const key = timelineEventKey(event);
            const prev = reversed[idx + 1];
            nextKeys.add(key);

            let article = existingByKey.get(key);
            const isNew = !article;
            if (!article) {
                article = createTimelineEvent();
            }
            updateTimelineEvent(article, event, prev);

            if (isNew && idx < 12) {
                article.classList.add('hud-event--enter');
                article.style.animationDelay = `${idx * 35}ms`;
                window.setTimeout(() => {
                    article.classList.remove('hud-event--enter');
                }, 420 + idx * 35);
            } else {
                article.style.animationDelay = '';
            }

            const currentChild = els.timeline.children[idx];
            if (currentChild !== article) {
                els.timeline.insertBefore(article, currentChild || null);
            }
        });

        Array.from(els.timeline.querySelectorAll('.hud-event')).forEach((card) => {
            if (!nextKeys.has(card.dataset.eventKey || '')) {
                card.remove();
            }
        });
    }

    function seedSupervisorControls(supervisor) {
        const config = supervisor.config || {};
        if (controlSeeded) return;
        els.piGoalInput.value = config.goal || '';
        els.piProviderInput.value = config.provider || '';
        els.piModelInput.value = config.model || '';
        els.piThinkingSelect.value = config.thinking || '';
        els.piAutoContinueInput.checked = Boolean(config.auto_continue ?? true);
        controlSeeded = true;
    }

    function renderSupervisor(supervisor) {
        const config = supervisor.config || {};
        seedSupervisorControls(supervisor);
        const isTurnActive = supervisor.status === 'starting' || supervisor.status === 'running';

        // capture session origin for relative-time labels
        const started = parseTs(supervisor.started_at);
        if (!Number.isNaN(started) && started) {
            sessionOriginMs = started;
        }

        const label = supervisor.available ? `PI ${supervisor.status || 'IDLE'}` : 'PI OFFLINE';
        setPiStatus(supervisor.status, label);
        els.piModelChip.textContent = supervisor.model
            ? `◉ PI: ${supervisor.model}`
            : `◉ PI: ${supervisor.pi_binary ? 'default model' : 'not installed'}`;
        els.piSessionChip.textContent = supervisor.session_id
            ? `SESSION: ${supervisor.session_id.slice(0, 8)}`
            : 'SESSION: NONE';
        els.piTurnsChip.textContent = `TURNS: ${supervisor.turns_completed || 0}`;
        els.piStatusSummary.textContent =
            supervisor.status_reason || supervisor.last_error || 'Pi supervisor standing by.';

        const counts = supervisor.counts || {};
        const sessionUsage = supervisor.session_usage || null;
        const compactionInfo = supervisor.compaction || {};
        const contextLabel = formatContextUsage(sessionUsage, supervisor.model_limits || null);
        renderKeyValueCards(els.piSupervisorStats, [
            ['STATUS', (supervisor.status || 'idle').toUpperCase()],
            ['MODEL', supervisor.model || 'default'],
            ['PROVIDER', supervisor.provider || 'default'],
            ['THINKING', supervisor.thinking || 'default'],
            ['AUTO', config.auto_continue ? 'on' : 'off'],
            ['DELAY', `${config.continue_delay_seconds ?? 1}s`],
            ['NEXT', supervisor.next_auto_continue_at ? timeLabel(supervisor.next_auto_continue_at) : 'n/a'],
            ['GOAL', truncate(config.goal || supervisor.goal || 'default loop', 24)],
            ['CTX', contextLabel],
            ['TOOL CALLS', formatCompactNumber(counts.tool_calls || 0)],
            ['THINK BLOCKS', formatCompactNumber(counts.thinking_blocks || 0)],
            ['AI MSGS', formatCompactNumber(counts.assistant_messages || 0)],
            ['USR MSGS', formatCompactNumber(counts.user_messages || 0)],
            [
                'LAST COMPACT',
                compactionInfo.tokens_before
                    ? `${formatCompactNumber(compactionInfo.tokens_before)}→${formatCompactNumber(compactionInfo.tokens_after || 0)}`
                    : 'none',
            ],
        ]);

        if (supervisor.last_error) {
            els.piControlStatus.textContent = `► LAST ERROR: ${supervisor.last_error}`;
        } else if (supervisor.next_auto_continue_at) {
            els.piControlStatus.textContent = `► AUTO-CONTINUE @ ${timeLabel(supervisor.next_auto_continue_at)}`;
        } else {
            els.piControlStatus.textContent = `► LAST EVENT: ${timeLabel(supervisor.last_event_at)}`;
        }

        const turnPlanPreview = supervisor.turn_plan_preview?.payload || supervisor.turn_plan_preview;
        els.piTurnPlanPreview.textContent = turnPlanPreview
            ? formatJSON(turnPlanPreview)
            : 'No Pi-authored turn plan captured yet.';

        const toolFeed = [
            ...(supervisor.active_tools || []),
            ...(supervisor.recent_tools || []).slice().reverse(),
        ];
        renderToolList(els.piToolFeed, toolFeed, 'No Pi tool calls yet.');
        renderTranscript(supervisor.transcript || []);
        renderList(
            els.piRecentEvents,
            (supervisor.recent_events || []).slice().reverse().map((event) => {
                return `${timeLabel(event.timestamp)} | ${event.type} | ${truncate(event.summary, 140)}`;
            }),
            'No recent Pi events.'
        );
        const followThinking = shouldAutoScroll('thinking', els.piCurrentThinking);
        if (els.piCurrentThinking.dataset.streaming !== 'true') {
            els.piCurrentThinking.textContent =
                supervisor.current_assistant_thinking ||
                (isTurnActive ? '' : supervisor.last_assistant_thinking) ||
                (isTurnActive ? '— waiting for comms —' : '— silent —');
            scrollNodeToBottom(els.piCurrentThinking, 'thinking', followThinking);
        }
        const followOutput = shouldAutoScroll('output', els.piCurrentOutput);
        if (els.piCurrentOutput.dataset.streaming !== 'true') {
            els.piCurrentOutput.textContent =
                supervisor.current_assistant_text ||
                (isTurnActive ? '' : supervisor.last_assistant_text) ||
                (isTurnActive ? '— waiting for comms —' : '— silent —');
            scrollNodeToBottom(els.piCurrentOutput, 'output', followOutput);
        }
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
        const planStatus = intent.plan_status || payload.plan_status || {};
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
        els.lastUpdate.textContent = `◉ ${timeLabel(payload.generated_at)}`;
        els.uiModeChip.textContent = `◉ UI: ${(visuals.ui_mode || 'unknown').toUpperCase()}`;
        els.realtimeChip.textContent = serverRuntime.realtime_enabled
            ? `◉ CLK: ${serverRuntime.realtime_fps || 60}/${serverRuntime.live_artifact_fps || 0} FPS`
            : '◉ CLK: PAUSED';
        els.frameTimestamp.textContent = timeLabel(visuals.frame_timestamp);
        els.screenTextSource.textContent = `SRC: ${(screenText.source || 'n/a').toUpperCase()}`;

        if (artifactUrls.latest_frame_annotated) {
            schedulePreload(
                'annotated',
                els.annotatedFrame,
                withCacheBust(artifactUrls.latest_frame_annotated, visuals.frame_timestamp),
            );
        }
        if (artifactUrls.latest_frame) {
            schedulePreload(
                'raw',
                els.rawFrame,
                withCacheBust(artifactUrls.latest_frame, visuals.frame_timestamp),
            );
        }
        const screenTextValue = screenText.text || 'No OCR or dialogue text available.';
        els.screenText.textContent = screenText.note
            ? `${screenTextValue}\n\n[${screenText.note}]`
            : screenTextValue;

        els.objectiveTitle.textContent = objective.title || 'No objective yet';
        els.objectiveProgress.textContent = `${objective.progress_percent ?? memory.progress_percent ?? 0}%`;
        els.objectiveSummary.textContent = objective.summary || 'No objective summary.';
        els.objectivePredicate.textContent = objective.completion_predicate || '';
        els.objectiveRoute.textContent = objective.route_hint || '';
        els.progressFill.style.width = `${memory.progress_percent ?? 0}%`;

        const turnPlanState = (planStatus.state || turnPlan.status?.state || 'awaiting_plan').replaceAll('_', ' ');
        els.turnPlanSummary.textContent = turnPlan.summary || `Plan status: ${turnPlanState}`;
        renderList(els.plannedActions, turnPlan.planned_actions, 'No planned actions set.');
        renderList(els.fallbackActions, turnPlan.fallback_actions, 'No fallback actions set.');
        els.turnPlanNotes.textContent =
            turnPlan.notes ||
            planStatus.reason ||
            `Plan status: ${turnPlanState} · updated ${turnPlan.updated_at || 'never'}`;

        els.recentActionSummary.textContent = recentAction.summary || 'No recent action summary.';
        renderList(els.recentActionNotes, recentAction.notes, 'No recent action notes.');
        renderList(els.stateDeltaSummary, stateDelta.summary, 'No state delta summary.');
        renderList(els.movementGuidance, movementGuidance.notes, 'No movement guidance available.');

        renderWorldStats(world, memory.progress_percent, serverRuntime);
        renderFrameHud(world, memory);
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
        if (!artifactUrls.turn_context_json) {
            els.rawObservation.textContent = 'No turn_context.json artifact available.';
            return;
        }
        try {
            const response = await fetch(
                withCacheBust(artifactUrls.turn_context_json, payload.generated_at)
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
            goal: els.piGoalInput.value.trim() || null,
            provider: els.piProviderInput.value.trim() || null,
            model: els.piModelInput.value.trim() || null,
            thinking: els.piThinkingSelect.value || null,
            auto_continue: els.piAutoContinueInput.checked,
        };
        resetLiveStreamView();
        scrollTranscriptToBottom(true);
        els.piControlStatus.textContent = '► STARTING PI…';
        try {
            await postJson('/supervisor/start', body);
            els.piControlStatus.textContent = '► PI SUPERVISOR ONLINE';
            await refreshAll();
        } catch (error) {
            els.piControlStatus.textContent = `► ${String(error.message || error)}`;
        }
    }

    async function continueSupervisor() {
        resetLiveStreamView();
        scrollTranscriptToBottom(true);
        els.piControlStatus.textContent = '► CONTINUING PI…';
        try {
            await postJson('/supervisor/continue', {});
            els.piControlStatus.textContent = '► MANUAL CONTINUE DISPATCHED';
            await refreshAll();
        } catch (error) {
            els.piControlStatus.textContent = `► ${String(error.message || error)}`;
        }
    }

    async function stopSupervisor() {
        els.piControlStatus.textContent = '► STOPPING PI…';
        try {
            await postJson('/supervisor/stop');
            els.piControlStatus.textContent = '► PI SUPERVISOR OFFLINE';
            await refreshAll();
        } catch (error) {
            els.piControlStatus.textContent = `► ${String(error.message || error)}`;
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
        const empty = !current || current === 'No stderr output.';
        els.piStderr.textContent = empty ? text : `${current}\n${text}`;
    }

    const framePreloaders = {
        annotated: { loader: null, latestUrl: null },
        raw: { loader: null, latestUrl: null },
    };

    function schedulePreload(kind, targetImg, url) {
        if (!url) return;
        const slot = framePreloaders[kind];
        slot.latestUrl = url;
        const loader = new Image();
        slot.loader = loader;
        loader.onload = () => {
            if (slot.latestUrl !== url) return;
            targetImg.src = url;
        };
        loader.onerror = () => {
            if (slot.latestUrl !== url) return;
            slot.loader = null;
        };
        loader.src = url;
    }

    function applyFrameUpdate(data) {
        if (!data || typeof data !== 'object') return;
        if (data.annotated_frame_url) {
            const url = withCacheBust(data.annotated_frame_url, data.frame_timestamp);
            schedulePreload('annotated', els.annotatedFrame, url);
        }
        if (data.raw_frame_url) {
            const url = withCacheBust(data.raw_frame_url, data.frame_timestamp);
            schedulePreload('raw', els.rawFrame, url);
        }
        if (data.frame_timestamp) {
            els.frameTimestamp.textContent = timeLabel(data.frame_timestamp);
        }
    }

    const streamBuffers = { output: '', thinking: '' };
    const streamPendingReset = { output: false, thinking: false };
    let streamRafPending = false;

    function resetLiveStreamView(placeholder = '— waiting for comms —') {
        streamBuffers.output = '';
        streamBuffers.thinking = '';
        streamPendingReset.output = false;
        streamPendingReset.thinking = false;
        autoScrollState.thinking = true;
        autoScrollState.output = true;
        els.piCurrentThinking.textContent = placeholder;
        els.piCurrentThinking.dataset.streaming = 'false';
        els.piCurrentOutput.textContent = placeholder;
        els.piCurrentOutput.dataset.streaming = 'false';
    }

    function flushStreamBuffers() {
        streamRafPending = false;
        const followOutput = shouldAutoScroll('output', els.piCurrentOutput);
        const followThinking = shouldAutoScroll('thinking', els.piCurrentThinking);

        if (streamPendingReset.output) {
            els.piCurrentOutput.textContent = '';
            streamPendingReset.output = false;
        }
        if (streamBuffers.output) {
            const last = els.piCurrentOutput.lastChild;
            if (last && last.nodeType === Node.TEXT_NODE) {
                last.textContent += streamBuffers.output;
            } else {
                els.piCurrentOutput.appendChild(document.createTextNode(streamBuffers.output));
            }
            streamBuffers.output = '';
            scrollNodeToBottom(els.piCurrentOutput, 'output', followOutput);
        }

        if (streamPendingReset.thinking) {
            els.piCurrentThinking.textContent = '';
            streamPendingReset.thinking = false;
        }
        if (streamBuffers.thinking) {
            const last = els.piCurrentThinking.lastChild;
            if (last && last.nodeType === Node.TEXT_NODE) {
                last.textContent += streamBuffers.thinking;
            } else {
                els.piCurrentThinking.appendChild(document.createTextNode(streamBuffers.thinking));
            }
            streamBuffers.thinking = '';
            scrollNodeToBottom(els.piCurrentThinking, 'thinking', followThinking);
        }
    }

    function scheduleStreamFlush() {
        if (streamRafPending) return;
        streamRafPending = true;
        requestAnimationFrame(flushStreamBuffers);
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
            resetLiveStreamView();
            scrollTranscriptToBottom();
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
                streamPendingReset.output = true;
                streamBuffers.output = event.text;
                els.piCurrentOutput.dataset.streaming = 'true';
            } else if (typeof event.delta === 'string' && event.delta) {
                if (els.piCurrentOutput.dataset.streaming !== 'true') {
                    streamPendingReset.output = true;
                }
                streamBuffers.output += event.delta;
                els.piCurrentOutput.dataset.streaming = 'true';
            }
            scheduleStreamFlush();
            return;
        }

        if (event.type === 'pi_thinking_delta') {
            if (typeof event.thinking === 'string' && event.thinking) {
                streamPendingReset.thinking = true;
                streamBuffers.thinking = event.thinking;
                els.piCurrentThinking.dataset.streaming = 'true';
            } else if (typeof event.delta === 'string' && event.delta) {
                if (els.piCurrentThinking.dataset.streaming !== 'true') {
                    streamPendingReset.thinking = true;
                }
                streamBuffers.thinking += event.delta;
                els.piCurrentThinking.dataset.streaming = 'true';
            }
            scheduleStreamFlush();
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
            resetLiveStreamView();
            scrollTranscriptToBottom();
            els.piControlStatus.textContent = `► ${truncate(event.summary || 'Pi turn launched.', 220)}`;
            scheduleRefresh(150);
            return;
        }

        if (event.type === 'pi_auto_continue_scheduled') {
            els.piControlStatus.textContent = event.summary || 'Auto-continue scheduled.';
            scheduleRefresh(150);
            return;
        }

        if (event.type === 'screenshot') {
            applyFrameUpdate(event.data || {});
            if ((event.data || {}).source !== 'live_sync') {
                scheduleRefresh(100);
            }
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
        setPiStatus('idle', 'PI IDLE');
        sessionOriginMs = Date.now();
        renderTimelineFilters({});
        initAutoScroll('transcript', els.piTranscript);
        initAutoScroll('thinking', els.piCurrentThinking);
        initAutoScroll('output', els.piCurrentOutput);
        els.piStartButton.addEventListener('click', startSupervisor);
        els.piContinueButton.addEventListener('click', continueSupervisor);
        els.piStopButton.addEventListener('click', stopSupervisor);
        els.manualSaveButton.addEventListener('click', saveNow);
        els.loadSaveButton.addEventListener('click', loadSelectedSave);
        els.loadRecommendedButton.addEventListener('click', loadRecommendedSave);
        const viewports = frameViewports();
        if (viewports.length) {
            syncFrameFullscreenState();
            viewports.forEach((viewport) => {
                viewport.addEventListener('click', onFrameViewportClick);
                viewport.addEventListener('keydown', onFrameViewportKeydown);
            });
            document.addEventListener('fullscreenchange', syncFrameFullscreenState);
            document.addEventListener('webkitfullscreenchange', syncFrameFullscreenState);
        }
        // Prompt textarea: expand on focus, collapse on blur-when-empty
        if (els.piGoalInput) {
            els.piGoalInput.addEventListener('focus', () => {
                els.piGoalInput.rows = 4;
            });
            els.piGoalInput.addEventListener('blur', () => {
                if (!els.piGoalInput.value.trim()) {
                    els.piGoalInput.rows = 1;
                }
            });
        }
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
