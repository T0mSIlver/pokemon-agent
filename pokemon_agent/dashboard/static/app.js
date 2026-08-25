(() => {
    'use strict';

    const POLL_INTERVAL = 3000;
    const HISTORY_LIMIT = 180;
    const STREAM_DOM_WINDOW = 500;
    const STREAM_WINDOW_STEP = 250;
    const STREAM_MEMORY_LIMIT = 5000;
    const STREAM_FETCH_LIMIT = 1000;
    const STREAM_FETCH_PAGES = 12;
    const STREAM_THINKING_LINES = 2;
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
        annotatedFrame: $('annotatedFrame'),
        rawFrame: $('rawFrame'),
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
        piSteerInput: $('piSteerInput'),
        piSteerButton: $('piSteerButton'),
        piSteerStatus: $('piSteerStatus'),
        piControlStatus: $('piControlStatus'),
        piTurnPlanPreview: $('piTurnPlanPreview'),
        critiqueStateChip: $('critiqueStateChip'),
        critiqueMetaChip: $('critiqueMetaChip'),
        critiqueActiveGoal: $('critiqueActiveGoal'),
        critiqueGoalSource: $('critiqueGoalSource'),
        critiqueNextGoal: $('critiqueNextGoal'),
        critiqueError: $('critiqueError'),
        critiqueText: $('critiqueText'),
        piStream: $('piStream'),
        piStreamList: $('piStreamList'),
        piStreamOlder: $('piStreamOlder'),
        piStreamJump: $('piStreamJump'),
        piStreamSearch: $('piStreamSearch'),
        piStreamFilters: $('piStreamFilters'),
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
        campaignRungChip: $('campaignRungChip'),
        campaignPressesChip: $('campaignPressesChip'),
        campaignSourceChip: $('campaignSourceChip'),
        campaignHeadline: $('campaignHeadline'),
        campaignStats: $('campaignStats'),
        campaignRail: $('campaignRail'),
        campaignRailFill: $('campaignRailFill'),
        campaignRailReadout: $('campaignRailReadout'),
        campaignChart: $('campaignChart'),
        campaignChartCaption: $('campaignChartCaption'),
        campaignBenchmark: $('campaignBenchmark'),
        campaignLadderDetails: $('campaignLadderDetails'),
        campaignLadderRows: $('campaignLadderRows'),
        healthWindowChip: $('healthWindowChip'),
        healthStrip: $('healthStrip'),
    };

    let ws = null;
    let wsReconnectDelay = WS_RECONNECT_BASE;
    let wsReconnectTimer = null;
    let pollTimer = null;
    let refreshTimer = null;
    let refreshInFlight = null;
    let controlSeeded = false;
    let steerLive = null;
    let latestRecovery = {};
    let latestSaves = [];
    let latestTimelineEvents = [];
    let sessionOriginMs = null;
    const autoScrollState = {
        stream: true,
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

    const STREAM_KINDS = new Set(['tool', 'thinking', 'text', 'user', 'system']);
    // Supervisor statuses where an operator message has a live session to land in.
    const STEERABLE_PI_STATUSES = new Set(['starting', 'running']);
    const STREAM_SYSTEM_LEVELS = new Set(['info', 'warn', 'error']);
    const STREAM_KIND_FILTERS = [
        { key: 'all',           label: 'ALL' },
        { key: 'kind:tool',     label: 'TOOL' },
        { key: 'kind:thinking', label: 'THINK' },
        { key: 'kind:text',     label: 'SAID' },
        { key: 'kind:user',     label: 'PROMPT' },
        { key: 'kind:system',   label: 'SYSTEM' },
        { key: 'state:error',   label: 'ERRORS' },
    ];
    const STREAM_TOOL_CHIP_LIMIT = 8;

    /* ------------------------------------------------------------------ */
    /* Campaign ladder                                                     */
    /* ------------------------------------------------------------------ */

    // The 63-rung Red ladder, mirrored from pokemon_agent/data/red_milestones.json.
    // The browser has no endpoint for that file and this page has no build step,
    // so it is embedded. tests/test_dashboard.py fails if the two drift apart.
    const RED_LADDER_RAW = [
        ["EVENT_GOT_STARTER", "Chose a starter Pokemon", "event"],
        ["EVENT_BATTLED_RIVAL_IN_OAKS_LAB", "Fought the rival in Oak's Lab", "event"],
        ["EVENT_GOT_OAKS_PARCEL", "Picked up Oak's Parcel in Viridian", "event"],
        ["EVENT_OAK_GOT_PARCEL", "Delivered Oak's Parcel", "event"],
        ["EVENT_GOT_POKEDEX", "Received the Pokedex", "event"],
        ["EVENT_GOT_TOWN_MAP", "Got the Town Map from Daisy", "event"],
        ["EVENT_BEAT_ROUTE22_RIVAL_1ST_BATTLE", "Beat the rival on Route 22", "event"],
        ["EVENT_BEAT_BROCK", "Defeated Brock", "event"],
        ["BADGE_BOULDER", "Boulder Badge", "badge"],
        ["EVENT_BEAT_MT_MOON_EXIT_SUPER_NERD", "Beat the Super Nerd guarding the Mt. Moon fossils", "event"],
        ["EVENT_GOT_DOME_FOSSIL", "Took the Dome Fossil", "event"],
        ["EVENT_GOT_HELIX_FOSSIL", "Took the Helix Fossil", "event"],
        ["EVENT_BEAT_CERULEAN_RIVAL", "Beat the rival in Cerulean City", "event"],
        ["EVENT_BEAT_MISTY", "Defeated Misty", "event"],
        ["BADGE_CASCADE", "Cascade Badge", "badge"],
        ["EVENT_MET_BILL", "Met Bill on Route 25", "event"],
        ["EVENT_GOT_SS_TICKET", "Got the S.S. Ticket", "event"],
        ["EVENT_RUBBED_CAPTAINS_BACK", "Cured the S.S. Anne captain", "event"],
        ["EVENT_GOT_HM01", "Got HM01 Cut", "event"],
        ["EVENT_SS_ANNE_LEFT", "The S.S. Anne set sail", "event"],
        ["EVENT_BEAT_LT_SURGE", "Defeated Lt. Surge", "event"],
        ["BADGE_THUNDER", "Thunder Badge", "badge"],
        ["EVENT_GOT_OLD_AMBER", "Got the Old Amber in the Pewter Museum", "event"],
        ["EVENT_GOT_BIKE_VOUCHER", "Got the Bike Voucher", "event"],
        ["EVENT_GOT_BICYCLE", "Got the Bicycle", "event"],
        ["EVENT_GOT_HM05", "Got HM05 Flash", "event"],
        ["EVENT_FOUND_ROCKET_HIDEOUT", "Found the Rocket Hideout under the Game Corner", "event"],
        ["ITEM_LIFT_KEY", "Have the Lift Key", "item"],
        ["EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI", "Defeated Giovanni in the Rocket Hideout", "event"],
        ["ITEM_SILPH_SCOPE", "Have the Silph Scope", "item"],
        ["EVENT_BEAT_ERIKA", "Defeated Erika", "event"],
        ["BADGE_RAINBOW", "Rainbow Badge", "badge"],
        ["EVENT_BEAT_POKEMON_TOWER_RIVAL", "Beat the rival in Pokemon Tower", "event"],
        ["EVENT_BEAT_GHOST_MAROWAK", "Laid the Marowak ghost to rest", "event"],
        ["EVENT_RESCUED_MR_FUJI", "Rescued Mr. Fuji", "event"],
        ["EVENT_GOT_POKE_FLUTE", "Got the Poke Flute", "event"],
        ["EVENT_BEAT_ROUTE12_SNORLAX", "Cleared the Snorlax on Route 12", "event"],
        ["EVENT_GOT_HM02", "Got HM02 Fly", "event"],
        ["EVENT_GOT_HM03", "Got HM03 Surf in the Safari Zone", "event"],
        ["EVENT_GAVE_GOLD_TEETH", "Returned the Gold Teeth to the Warden", "event"],
        ["EVENT_GOT_HM04", "Got HM04 Strength", "event"],
        ["EVENT_BEAT_KOGA", "Defeated Koga", "event"],
        ["BADGE_SOUL", "Soul Badge", "badge"],
        ["ITEM_CARD_KEY", "Have the Card Key", "item"],
        ["EVENT_BEAT_SILPH_CO_RIVAL", "Beat the rival in Silph Co.", "event"],
        ["EVENT_BEAT_SILPH_CO_GIOVANNI", "Defeated Giovanni in Silph Co.", "event"],
        ["EVENT_GOT_MASTER_BALL", "Got the Master Ball", "event"],
        ["EVENT_BEAT_SABRINA", "Defeated Sabrina", "event"],
        ["BADGE_MARSH", "Marsh Badge", "badge"],
        ["ITEM_SECRET_KEY", "Have the Secret Key", "item"],
        ["EVENT_BEAT_BLAINE", "Defeated Blaine", "event"],
        ["BADGE_VOLCANO", "Volcano Badge", "badge"],
        ["EVENT_BEAT_VIRIDIAN_GYM_GIOVANNI", "Defeated Giovanni in the Viridian Gym", "event"],
        ["BADGE_EARTH", "Earth Badge", "badge"],
        ["EVENT_BEAT_ROUTE22_RIVAL_2ND_BATTLE", "Beat the rival on Route 22 again", "event"],
        ["EVENT_VICTORY_ROAD_2_BOULDER_ON_SWITCH2", "Opened the Victory Road 2F barrier", "event"],
        ["EVENT_AUTOWALKED_INTO_LORELEIS_ROOM", "Cleared Victory Road and reached the Elite Four", "event"],
        ["EVENT_BEAT_LORELEIS_ROOM_TRAINER_0", "Defeated Lorelei", "event"],
        ["EVENT_BEAT_BRUNOS_ROOM_TRAINER_0", "Defeated Bruno", "event"],
        ["EVENT_BEAT_AGATHAS_ROOM_TRAINER_0", "Defeated Agatha", "event"],
        ["EVENT_BEAT_LANCE", "Defeated Lance", "event"],
        ["EVENT_BEAT_CHAMPION_RIVAL", "Defeated the Champion", "event"],
        ["EVENT_HALL_OF_FAME_DEX_RATING", "Entered the Hall of Fame", "event"],
    ];
    const RED_LADDER = RED_LADDER_RAW.map(([id, label, kind], index) => ({ id, label, kind, index }));
    const LADDER_BY_ID = new Map(RED_LADDER.map((rung) => [rung.id, rung]));
    const LADDER_BY_LABEL = new Map(RED_LADDER.map((rung) => [rung.label.toLowerCase(), rung]));
    const FIRST_GYM_ID = 'EVENT_BEAT_BROCK';

    // Published reference points, the same three the bench report quotes.
    const REF_POKEAGENT_BEST = 1608;
    const REF_POKEAGENT_EFFICIENT = 649;
    const REF_HUMAN_SPEEDRUN_SECONDS = 18 * 60;

    const LEDGER_STORAGE_KEY = 'poke-dash-press-ledger:v2';
    const PRESS_SAMPLE_LIMIT = 90;
    const PROGRESS_RETRY_INTERVAL = 30000;

    const HEALTH_GLYPH = { good: '\u25cf', warn: '\u25b2', crit: '\u2715', idle: '\u00b7' };

    // value >= crit is critical, value >= warn is a warning, below both is fine.
    const HEALTH_SPECS = [
        { key: 'blocked',   label: 'Blocked',   warn: 0.10, crit: 0.25 },
        { key: 'toolerr',   label: 'Tool err',  warn: 0.02, crit: 0.08 },
        { key: 'revisit',   label: 'Revisit',   warn: 2.0,  crit: 3.5  },
        { key: 'whiteouts', label: 'Whiteouts', warn: 1,    crit: 3    },
        { key: 'reloads',   label: 'Reloads',   warn: 2,    crit: 5    },
    ];
    const streamFilters = new Set(['all']);
    let streamSearchQuery = '';

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

    function toolText(value, fallback) {
        const text = String(value ?? '').trim();
        return text || fallback;
    }

    function looksLikeSerializedJSON(text) {
        if (typeof text !== 'string') return false;
        const trimmed = text.trim();
        if (!trimmed) return false;
        const first = trimmed[0];
        const last = trimmed[trimmed.length - 1];
        return (
            (first === '{' && last === '}') ||
            (first === '[' && last === ']') ||
            (first === '"' && last === '"')
        );
    }

    function decodeToolPayloadValue(value, maxDepth = 2) {
        let current = value;
        for (let depth = 0; depth < maxDepth; depth += 1) {
            if (typeof current !== 'string') break;
            const trimmed = current.trim();
            if (!looksLikeSerializedJSON(trimmed)) break;
            try {
                current = JSON.parse(trimmed);
            } catch (_) {
                break;
            }
        }
        return current;
    }

    function isStructuredToolPayload(value) {
        return Array.isArray(value) || Boolean(value && typeof value === 'object');
    }

    function formatToolScalar(value) {
        const decoded = decodeToolPayloadValue(value, 1);
        if (decoded === null) return 'null';
        if (decoded === undefined) return '';
        if (typeof decoded === 'string') return decoded;
        if (typeof decoded === 'number' || typeof decoded === 'boolean') return String(decoded);
        try {
            return JSON.stringify(decoded, null, 2);
        } catch (_) {
            return String(decoded);
        }
    }

    function createToolPayloadText(text, muted = false) {
        const el = document.createElement('div');
        el.className = 'hud-tool-payload-text';
        if (muted) el.dataset.empty = 'true';
        el.textContent = text;
        return el;
    }

    function renderToolPayloadTree(value) {
        const decoded = decodeToolPayloadValue(value);
        if (!isStructuredToolPayload(decoded)) {
            return createToolPayloadText(formatToolScalar(decoded));
        }

        const tree = document.createElement('div');
        tree.className = 'hud-tool-payload-tree';
        const entries = Array.isArray(decoded)
            ? decoded.map((item, index) => [String(index), item])
            : Object.entries(decoded);

        if (!entries.length) {
            tree.appendChild(createToolPayloadText(Array.isArray(decoded) ? '[]' : '{}', true));
            return tree;
        }

        entries.forEach(([label, entryValue]) => {
            const row = document.createElement('div');
            row.className = 'hud-tool-payload-row';

            const key = document.createElement('div');
            key.className = 'hud-tool-payload-key';
            key.textContent = label;

            const val = document.createElement('div');
            val.className = 'hud-tool-payload-value';

            const normalized = decodeToolPayloadValue(entryValue);
            if (isStructuredToolPayload(normalized)) {
                const meta = document.createElement('div');
                meta.className = 'hud-tool-payload-meta';
                meta.textContent = Array.isArray(normalized)
                    ? `${normalized.length} item${normalized.length === 1 ? '' : 's'}`
                    : `${Object.keys(normalized).length} field${Object.keys(normalized).length === 1 ? '' : 's'}`;
                val.appendChild(meta);
                val.appendChild(renderToolPayloadTree(normalized));
            } else {
                val.appendChild(createToolPayloadText(formatToolScalar(normalized)));
            }

            row.appendChild(key);
            row.appendChild(val);
            tree.appendChild(row);
        });

        return tree;
    }

    function renderToolPayload(node, value, fallback) {
        if (!node) return;
        const decoded = decodeToolPayloadValue(value);
        if (decoded === null || decoded === undefined || decoded === '') {
            node.replaceChildren(createToolPayloadText(fallback, true));
            return;
        }
        if (isStructuredToolPayload(decoded)) {
            node.replaceChildren(renderToolPayloadTree(decoded));
            return;
        }
        node.replaceChildren(createToolPayloadText(formatToolScalar(decoded)));
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

    /* ------------------------------------------------------------------ */
    /* Merged live stream                                                  */
    /* ------------------------------------------------------------------ */

    const streamEntries = [];
    const streamBySeq = new Map();
    const streamOpenState = new Map();
    const streamDirtySeqs = new Set();
    let streamSessionId = null;
    let streamNextSeq = 0;
    let streamRenderLimit = STREAM_DOM_WINDOW;
    let streamPendingNew = 0;
    let streamRenderPending = false;
    let streamFlushPending = false;
    let streamChromeTimer = null;
    let streamSyncInFlight = null;
    let streamUnavailable = false;
    let streamExpanding = false;

    function streamOpen(seq) {
        let state = streamOpenState.get(seq);
        if (!state) {
            state = { cmd: false, res: false, more: false, touched: false };
            streamOpenState.set(seq, state);
        }
        return state;
    }

    function streamEntryKind(entry) {
        const kind = String(entry?.kind || 'text').toLowerCase();
        return STREAM_KINDS.has(kind) ? kind : 'text';
    }

    function streamEntryState(entry) {
        const state = String(entry?.state || 'ok').toLowerCase();
        if (state === 'running' || state === 'error') return state;
        return 'ok';
    }

    function streamToolName(entry) {
        return String(entry?.tool?.name || '').toLowerCase();
    }

    function streamEntryText(entry) {
        return typeof entry?.text === 'string' ? entry.text : '';
    }

    function streamHeadline(entry) {
        const tool = entry?.tool || {};
        const headline = String(tool.headline || '').trim();
        if (headline) return headline;
        const text = streamEntryText(entry).trim();
        if (text) return text;
        return String(tool.name || 'tool call');
    }

    function streamHaystack(entry) {
        const tool = entry?.tool || {};
        return [
            entry?.kind || '',
            entry?.state || '',
            entry?.text || '',
            tool.name || '',
            tool.headline || '',
            tool.command || '',
            tool.path || '',
            tool.result_summary || '',
            tool.result_full || '',
            entry?.system?.label || '',
        ]
            .join(' ')
            .toLowerCase();
    }

    function streamStateLabel(state) {
        if (state === 'running') return 'LIVE';
        if (state === 'error') return 'ERR';
        return 'OK';
    }

    function streamDurationLabel(ms) {
        if (typeof ms !== 'number' || !Number.isFinite(ms) || ms < 0) return '';
        if (ms < 1000) return `${Math.round(ms)}ms`;
        if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
        const minutes = Math.floor(ms / 60000);
        const seconds = Math.round((ms % 60000) / 1000);
        return `${minutes}m${String(seconds).padStart(2, '0')}s`;
    }

    function streamClockLabel(ts) {
        const t = parseTs(ts);
        if (Number.isNaN(t)) return '--:--:--';
        return new Date(t).toLocaleTimeString([], { hour12: false });
    }

    function buildStreamPredicate() {
        const query = streamSearchQuery.toLowerCase().trim();
        const showAll = streamFilters.has('all') || streamFilters.size === 0;
        const kinds = new Set();
        const names = new Set();
        let errorsOnly = false;
        if (!showAll) {
            streamFilters.forEach((key) => {
                if (key === 'state:error') errorsOnly = true;
                else if (key.startsWith('kind:')) kinds.add(key.slice(5));
                else if (key.startsWith('tool:')) names.add(key.slice(5));
            });
        }
        return (entry) => {
            if (!showAll) {
                if (errorsOnly && streamEntryState(entry) !== 'error') return false;
                if (kinds.size || names.size) {
                    const kindOk = kinds.size ? kinds.has(streamEntryKind(entry)) : false;
                    const nameOk = names.size ? names.has(streamToolName(entry)) : false;
                    if (!kindOk && !nameOk) return false;
                }
            }
            if (query && !streamHaystack(entry).includes(query)) return false;
            return true;
        };
    }

    function streamFilterCounts() {
        const counts = { all: streamEntries.length, 'state:error': 0 };
        const tools = new Map();
        streamEntries.forEach((entry) => {
            const kindKey = `kind:${streamEntryKind(entry)}`;
            counts[kindKey] = (counts[kindKey] || 0) + 1;
            if (streamEntryState(entry) === 'error') counts['state:error'] += 1;
            const name = streamToolName(entry);
            if (name) {
                counts[`tool:${name}`] = (counts[`tool:${name}`] || 0) + 1;
                tools.set(name, (tools.get(name) || 0) + 1);
            }
        });
        const toolKeys = Array.from(tools.entries())
            .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
            .slice(0, STREAM_TOOL_CHIP_LIMIT)
            .map(([name]) => name);
        return { counts, toolKeys };
    }

    function renderStreamFilters() {
        if (!els.piStreamFilters) return;
        const { counts, toolKeys } = streamFilterCounts();
        const chips = STREAM_KIND_FILTERS.slice();
        toolKeys.forEach((name) => {
            chips.push({ key: `tool:${name}`, label: name });
        });
        streamFilters.forEach((key) => {
            if (key.startsWith('tool:') && !chips.some((chip) => chip.key === key)) {
                chips.push({ key, label: key.slice(5) });
            }
        });

        const fragment = document.createDocumentFragment();
        chips.forEach(({ key, label }) => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'hud-tool-filter-chip';
            const active = streamFilters.has(key) || (streamFilters.size === 0 && key === 'all');
            chip.dataset.active = active ? 'true' : 'false';
            chip.dataset.filter = key;
            chip.innerHTML = `${escapeHtml(label)}<span class="hud-tool-filter-count">${counts[key] || 0}</span>`;
            chip.addEventListener('click', () => toggleStreamFilter(key));
            fragment.appendChild(chip);
        });
        els.piStreamFilters.replaceChildren(fragment);
    }

    function toggleStreamFilter(key) {
        if (key === 'all') {
            streamFilters.clear();
            streamFilters.add('all');
        } else {
            streamFilters.delete('all');
            if (streamFilters.has(key)) {
                streamFilters.delete(key);
            } else {
                streamFilters.add(key);
            }
            if (!streamFilters.size) streamFilters.add('all');
        }
        streamRenderLimit = STREAM_DOM_WINDOW;
        renderStream();
    }

    function streamMetaNode() {
        const meta = document.createElement('span');
        meta.className = 'hud-stream-meta';

        const tool = document.createElement('span');
        tool.className = 'hud-stream-tool';
        tool.dataset.slot = 'tool';
        tool.hidden = true;

        const duration = document.createElement('span');
        duration.className = 'hud-stream-dur';
        duration.dataset.slot = 'dur';
        duration.hidden = true;

        const state = document.createElement('span');
        state.className = 'hud-chip hud-chip--tiny hud-stream-state';
        state.dataset.slot = 'state';

        meta.append(tool, duration, state);
        return meta;
    }

    function buildStreamToolBody(main, seq) {
        const head = document.createElement('div');
        head.className = 'hud-stream-head';
        const headline = document.createElement('span');
        headline.className = 'hud-stream-headline';
        headline.dataset.slot = 'headline';
        head.append(headline, streamMetaNode());

        const figure = document.createElement('div');
        figure.className = 'hud-frame-view hud-stream-shot';
        figure.dataset.slot = 'figure';
        figure.dataset.frameLabel = 'frame read';
        figure.dataset.fullscreen = 'false';
        figure.setAttribute('role', 'button');
        figure.tabIndex = 0;
        figure.title = 'Click to toggle fullscreen';
        figure.setAttribute('aria-label', 'Toggle fullscreen frame view');
        figure.hidden = true;
        const shot = document.createElement('img');
        shot.dataset.slot = 'shot';
        shot.alt = 'Frame read';
        shot.loading = 'lazy';
        const shotTag = document.createElement('span');
        shotTag.className = 'hud-frame-label hud-frame-label--tr';
        shotTag.dataset.slot = 'shotTag';
        figure.append(shot, shotTag);
        figure.addEventListener('click', onFrameViewportClick);
        figure.addEventListener('keydown', onFrameViewportKeydown);

        const fold = document.createElement('details');
        fold.className = 'hud-details hud-stream-fold';
        fold.dataset.slot = 'cmdFold';
        fold.hidden = true;
        const summary = document.createElement('summary');
        const glyph = document.createElement('span');
        glyph.className = 'hud-details-glyph';
        glyph.textContent = '▸';
        const summaryLabel = document.createElement('span');
        summaryLabel.dataset.slot = 'cmdLabel';
        summaryLabel.textContent = 'command';
        summary.append(glyph, summaryLabel);
        const command = document.createElement('pre');
        command.className = 'hud-stream-code';
        command.dataset.slot = 'command';
        fold.append(summary, command);
        fold.addEventListener('toggle', () => {
            streamOpen(seq).cmd = fold.open;
        });

        const result = document.createElement('div');
        result.className = 'hud-stream-result';
        result.dataset.slot = 'resWrap';
        result.hidden = true;
        const resultToggle = document.createElement('button');
        resultToggle.type = 'button';
        resultToggle.className = 'hud-stream-result-line';
        resultToggle.dataset.slot = 'resToggle';
        resultToggle.setAttribute('aria-expanded', 'false');
        const arrow = document.createElement('span');
        arrow.className = 'hud-stream-arrow';
        arrow.textContent = '→';
        const resultSummary = document.createElement('span');
        resultSummary.dataset.slot = 'resSummary';
        resultToggle.append(arrow, resultSummary);
        const resultFull = document.createElement('div');
        resultFull.className = 'hud-tool-payload hud-stream-payload';
        resultFull.dataset.slot = 'resFull';
        resultFull.hidden = true;
        resultToggle.addEventListener('click', () => {
            const state = streamOpen(seq);
            state.res = !state.res;
            state.touched = true;
            applyStreamResultOpen(main, state.res);
        });
        result.append(resultToggle, resultFull);

        main.append(head, figure, fold, result);
    }

    function applyStreamResultOpen(scope, open) {
        const toggle = scope.querySelector('[data-slot="resToggle"]');
        const full = scope.querySelector('[data-slot="resFull"]');
        const expanded = Boolean(open) && Boolean(full?.dataset.has);
        if (toggle) toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        if (full) full.hidden = !expanded;
    }

    function streamEntrySource(entry) {
        const source = String(entry?.source || 'agent').toLowerCase();
        return source || 'agent';
    }

    function buildStreamProseBody(main, seq, kind) {
        const head = document.createElement('div');
        head.className = 'hud-stream-head';
        const kicker = document.createElement('span');
        kicker.className = 'hud-stream-kicker';
        kicker.textContent = kind === 'thinking' ? 'thinking' : kind === 'user' ? 'prompt' : 'said';
        kicker.dataset.slot = 'kicker';
        head.append(kicker, streamMetaNode());

        const body = document.createElement('pre');
        body.className = kind === 'thinking' ? 'hud-stream-think' : 'hud-stream-say';
        const text = document.createElement('span');
        text.dataset.slot = 'text';
        body.appendChild(text);
        if (kind !== 'user') {
            const caret = document.createElement('span');
            caret.className = 'hud-cursor';
            caret.setAttribute('aria-hidden', 'true');
            body.appendChild(caret);
        }

        main.append(head, body);

        if (kind === 'thinking') {
            const more = document.createElement('button');
            more.type = 'button';
            more.className = 'hud-stream-more';
            more.dataset.slot = 'more';
            more.setAttribute('aria-expanded', 'false');
            more.hidden = true;
            more.addEventListener('click', () => {
                const state = streamOpen(seq);
                state.more = !state.more;
                const entry = streamBySeq.get(seq);
                const node = main.parentElement;
                if (entry && node) updateStreamNode(node, entry);
            });
            main.appendChild(more);
        }
    }

    // A system entry is not only its label: 'critique ready' carries the whole
    // retrospective in its text, and that used to be rendered nowhere at all.
    function buildStreamSystemBody(main, seq) {
        const divider = document.createElement('div');
        divider.className = 'hud-stream-divider';
        divider.dataset.slot = 'divider';
        divider.dataset.level = 'info';
        const label = document.createElement('span');
        label.className = 'hud-stream-divider-label';
        label.dataset.slot = 'label';
        divider.appendChild(label);
        main.appendChild(divider);

        const body = document.createElement('pre');
        body.className = 'hud-stream-say hud-stream-system-text';
        body.dataset.slot = 'sysText';
        body.hidden = true;
        main.appendChild(body);

        const more = document.createElement('button');
        more.type = 'button';
        more.className = 'hud-stream-more';
        more.dataset.slot = 'sysMore';
        more.setAttribute('aria-expanded', 'false');
        more.hidden = true;
        more.addEventListener('click', () => {
            const state = streamOpen(seq);
            state.more = !state.more;
            const entry = streamBySeq.get(seq);
            const node = main.parentElement;
            if (entry && node) updateStreamNode(node, entry);
        });
        main.appendChild(more);
    }

    function createStreamNode(entry) {
        const kind = streamEntryKind(entry);
        const seq = Number(entry.seq);

        const article = document.createElement('article');
        article.className = 'hud-stream-entry';
        article.dataset.kind = kind;
        article.dataset.seq = String(seq);

        const gutter = document.createElement('div');
        gutter.className = 'hud-stream-gutter';
        const time = document.createElement('time');
        time.className = 'hud-chat-time';
        time.dataset.slot = 'time';
        gutter.appendChild(time);

        const main = document.createElement('div');
        main.className = 'hud-stream-main';

        if (kind === 'tool') {
            buildStreamToolBody(main, seq);
        } else if (kind === 'system') {
            buildStreamSystemBody(main, seq);
        } else {
            buildStreamProseBody(main, seq, kind);
        }

        article.append(gutter, main);
        updateStreamNode(article, entry);
        return article;
    }

    function updateStreamNode(node, entry) {
        if (!node || !entry) return;
        const kind = streamEntryKind(entry);
        const state = streamEntryState(entry);
        const seq = Number(entry.seq);
        node.dataset.state = state;

        node.dataset.source = streamEntrySource(entry);
        const kicker = node.querySelector('[data-slot="kicker"]');
        if (kicker && kind === 'user') {
            kicker.textContent = node.dataset.source === 'operator' ? 'operator' : 'prompt';
        }

        const time = node.querySelector('[data-slot="time"]');
        if (time) {
            time.textContent = streamClockLabel(entry.ts);
            time.dateTime = entry.ts || '';
            time.title = timeLabel(entry.ts);
        }

        const stateChip = node.querySelector('[data-slot="state"]');
        if (stateChip) {
            stateChip.hidden = kind !== 'tool' && state === 'ok';
            stateChip.textContent = streamStateLabel(state);
            stateChip.dataset.status = state;
        }

        const duration = node.querySelector('[data-slot="dur"]');
        if (duration) {
            const label = streamDurationLabel(entry?.tool?.duration_ms);
            duration.textContent = label;
            duration.hidden = !label;
        }

        const toolTag = node.querySelector('[data-slot="tool"]');
        if (toolTag) {
            const name = String(entry?.tool?.name || '');
            toolTag.textContent = name;
            toolTag.hidden = !name;
        }

        if (kind === 'tool') {
            updateStreamToolNode(node, entry, state, seq);
        } else if (kind === 'system') {
            updateStreamSystemNode(node, entry, seq);
        } else {
            updateStreamProseNode(node, entry, kind === 'thinking', seq);
        }
    }

    function updateStreamToolNode(node, entry, state, seq) {
        const tool = entry.tool || {};
        node.dataset.tool = streamToolName(entry);

        const headline = node.querySelector('[data-slot="headline"]');
        if (headline) headline.textContent = streamHeadline(entry);

        const figure = node.querySelector('[data-slot="figure"]');
        const shot = node.querySelector('[data-slot="shot"]');
        const artifact = String(tool.image_artifact || '').trim();
        if (figure && shot) {
            if (artifact) {
                const url = withCacheBust(
                    `/artifacts/${encodeURIComponent(artifact)}`,
                    `${entry.seq}-${entry.ts || ''}`
                );
                if (shot.dataset.url !== url) {
                    shot.dataset.url = url;
                    shot.src = url;
                }
                const shotTag = node.querySelector('[data-slot="shotTag"]');
                if (shotTag) shotTag.textContent = artifact.replace(/_/g, ' ').toUpperCase();
                figure.dataset.frameLabel = `${artifact.replace(/_/g, ' ')} frame`;
                figure.hidden = false;
            } else {
                figure.hidden = true;
            }
        }

        const fold = node.querySelector('[data-slot="cmdFold"]');
        const command = node.querySelector('[data-slot="command"]');
        const commandText = String(tool.command || '').trim();
        const pathText = String(tool.path || '').trim();
        const foldText = commandText || pathText;
        if (fold && command) {
            if (foldText) {
                if (command.textContent !== foldText) command.textContent = foldText;
                const label = node.querySelector('[data-slot="cmdLabel"]');
                if (label) label.textContent = commandText ? 'command' : 'path';
                fold.hidden = false;
            } else {
                fold.hidden = true;
            }
        }

        const open = streamOpen(seq);
        const resultWrap = node.querySelector('[data-slot="resWrap"]');
        const resultSummary = node.querySelector('[data-slot="resSummary"]');
        const resultFull = node.querySelector('[data-slot="resFull"]');
        const summaryText = String(tool.result_summary || '').trim();
        const fullText = tool.result_full;
        const hasFull = fullText !== undefined && fullText !== null && String(fullText).trim() !== '';

        if (resultWrap && resultSummary && resultFull) {
            const fallback = state === 'running'
                ? 'waiting for output…'
                : hasFull
                    ? 'output captured'
                    : '';
            const line = summaryText || fallback;
            if (line || hasFull) {
                resultSummary.textContent = truncate(line.replace(/\s+/g, ' ').trim(), 200) || 'output';
                resultWrap.hidden = false;
            } else {
                resultWrap.hidden = true;
            }

            if (hasFull) {
                const signature = `${String(fullText).length}:${state}`;
                if (resultFull.dataset.signature !== signature) {
                    resultFull.dataset.signature = signature;
                    renderToolPayload(resultFull, fullText, 'No output captured.');
                }
                resultFull.dataset.has = '1';
            } else {
                delete resultFull.dataset.has;
                delete resultFull.dataset.signature;
                resultFull.replaceChildren();
            }

            if (state === 'error' && !open.touched) {
                open.res = hasFull;
                open.cmd = Boolean(foldText);
            }
            applyStreamResultOpen(node, open.res);
        }

        if (fold && !fold.hidden && fold.open !== Boolean(open.cmd)) {
            fold.open = Boolean(open.cmd);
        }
    }

    // Two lines collapsed, every character when expanded. The pre that holds the
    // text takes data-expanded so CSS can give it its own scroll instead of letting
    // a long critique stretch the panel.
    function applyExpandableText(slot, more, text, expanded) {
        if (!slot) return;
        const head = text.split('\n').slice(0, STREAM_THINKING_LINES).join('\n');
        const hiddenChars = Math.max(0, text.length - head.length);
        const open = Boolean(expanded) && hiddenChars > 0;
        const shown = open || hiddenChars === 0 ? text : head;
        if (slot.textContent !== shown) slot.textContent = shown;
        const box = slot.closest('pre') || slot;
        box.dataset.expanded = open ? 'true' : 'false';
        if (!more) return;
        if (hiddenChars > 0) {
            more.hidden = false;
            more.textContent = open
                ? '▾ collapse'
                : `▸ ${hiddenChars.toLocaleString()} more character${hiddenChars === 1 ? '' : 's'}`;
            more.setAttribute('aria-expanded', open ? 'true' : 'false');
        } else {
            more.hidden = true;
        }
    }

    function updateStreamProseNode(node, entry, collapsible, seq) {
        const text = streamEntryText(entry);
        const slot = node.querySelector('[data-slot="text"]');
        const more = node.querySelector('[data-slot="more"]');
        if (!slot) return;

        if (!collapsible) {
            if (slot.textContent !== text) slot.textContent = text;
            if (more) more.hidden = true;
            return;
        }

        applyExpandableText(slot, more, text, streamOpen(seq).more);
    }

    function updateStreamSystemNode(node, entry, seq) {
        const system = entry.system || {};
        const level = String(system.level || 'info').toLowerCase();
        const divider = node.querySelector('[data-slot="divider"]');
        const label = node.querySelector('[data-slot="label"]');
        const labelText = String(system.label || streamEntryText(entry) || 'event').trim();
        if (divider) {
            divider.dataset.level = STREAM_SYSTEM_LEVELS.has(level) ? level : 'info';
        }
        if (label) {
            label.textContent = `▶ ${labelText}`;
        }

        const body = node.querySelector('[data-slot="sysText"]');
        const more = node.querySelector('[data-slot="sysMore"]');
        const text = streamEntryText(entry).trim();
        // The label is often the whole story; only show a body when it is not.
        const hasBody = Boolean(text) && text !== labelText;
        if (body) body.hidden = !hasBody;
        if (!hasBody) {
            if (more) more.hidden = true;
            return;
        }
        applyExpandableText(body, more, text, streamOpen(seq).more);
    }

    function streamInsertIndex(seq) {
        let lo = 0;
        let hi = streamEntries.length;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (Number(streamEntries[mid].seq) < seq) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    function ingestStreamEntry(raw) {
        if (!raw || typeof raw !== 'object') return null;
        const seq = Number(raw.seq);
        if (!Number.isFinite(seq)) return null;
        if (seq + 1 > streamNextSeq) streamNextSeq = seq + 1;

        const existing = streamBySeq.get(seq);
        if (existing) {
            Object.keys(existing).forEach((key) => {
                if (!(key in raw)) delete existing[key];
            });
            Object.assign(existing, raw);
            return { entry: existing, isNew: false };
        }

        const entry = { ...raw };
        const last = streamEntries[streamEntries.length - 1];
        if (!last || Number(last.seq) < seq) {
            streamEntries.push(entry);
        } else {
            streamEntries.splice(streamInsertIndex(seq), 0, entry);
        }
        streamBySeq.set(seq, entry);

        while (streamEntries.length > STREAM_MEMORY_LIMIT) {
            const dropped = streamEntries.shift();
            streamBySeq.delete(Number(dropped.seq));
            streamOpenState.delete(Number(dropped.seq));
        }
        return { entry, isNew: true };
    }

    function resetStream() {
        streamEntries.length = 0;
        streamBySeq.clear();
        streamOpenState.clear();
        streamDirtySeqs.clear();
        streamNextSeq = 0;
        streamPendingNew = 0;
        streamRenderLimit = STREAM_DOM_WINDOW;
        autoScrollState.stream = true;
        if (els.piStreamList) els.piStreamList.replaceChildren();
    }

    function streamNodeFor(seq) {
        if (!els.piStreamList) return null;
        return els.piStreamList.querySelector(`[data-seq="${seq}"]`);
    }

    function scrollStreamToBottom() {
        scrollNodeToBottom(els.piStream, 'stream', true);
    }

    function captureStreamAnchor() {
        const viewport = els.piStream;
        const list = els.piStreamList;
        if (!viewport || !list || autoScrollState.stream) return null;
        const top = viewport.scrollTop;
        for (const child of list.children) {
            if (child.offsetTop + child.offsetHeight >= top) {
                return { seq: child.dataset.seq, offset: child.offsetTop - top };
            }
        }
        return null;
    }

    function restoreStreamAnchor(anchor) {
        const viewport = els.piStream;
        if (!anchor || !viewport) return;
        const node = streamNodeFor(anchor.seq);
        if (!node) return;
        viewport.scrollTop = node.offsetTop - anchor.offset;
    }

    function updateStreamJump() {
        const button = els.piStreamJump;
        if (!button) return;
        const following = autoScrollState.stream;
        button.hidden = following;
        if (following) return;
        button.textContent = streamPendingNew > 0
            ? `↓ ${streamPendingNew.toLocaleString()} new ${streamPendingNew === 1 ? 'entry' : 'entries'} · jump to live`
            : '↓ jump to live';
    }

    function refreshStreamChrome(visibleCount) {
        const count = typeof visibleCount === 'number'
            ? visibleCount
            : streamEntries.filter(buildStreamPredicate()).length;
        const hidden = Math.max(0, count - streamRenderLimit);
        if (els.piStreamOlder) {
            els.piStreamOlder.hidden = hidden === 0;
            if (hidden > 0) {
                els.piStreamOlder.textContent =
                    `▲ ${hidden.toLocaleString()} older ${hidden === 1 ? 'entry' : 'entries'} · load more`;
            }
        }
        renderStreamFilters();
        updateStreamJump();
    }

    function scheduleStreamChrome() {
        if (streamChromeTimer) return;
        streamChromeTimer = window.setTimeout(() => {
            streamChromeTimer = null;
            refreshStreamChrome();
        }, 500);
    }

    function scheduleStreamRender() {
        if (streamRenderPending) return;
        streamRenderPending = true;
        requestAnimationFrame(() => {
            streamRenderPending = false;
            renderStream();
        });
    }

    function renderStreamEmpty(message) {
        const empty = document.createElement('p');
        empty.className = 'hud-empty';
        empty.textContent = message;
        els.piStreamList.replaceChildren(empty);
    }

    function renderStream() {
        const list = els.piStreamList;
        const viewport = els.piStream;
        if (!list || !viewport) return;

        const follow = autoScrollState.stream;
        const anchor = captureStreamAnchor();
        const visible = streamEntries.filter(buildStreamPredicate());

        if (!visible.length) {
            renderStreamEmpty(
                streamEntries.length
                    ? '[ NO MATCHING ENTRIES ]'
                    : streamUnavailable
                        ? '[ STREAM UNAVAILABLE ]'
                        : 'AWAITING TRANSMISSION…'
            );
            refreshStreamChrome(0);
            return;
        }

        const hidden = Math.max(0, visible.length - streamRenderLimit);
        const windowEntries = hidden ? visible.slice(hidden) : visible;

        const emptyNode = list.querySelector('.hud-empty');
        if (emptyNode) emptyNode.remove();

        const existing = new Map();
        Array.from(list.children).forEach((child) => {
            existing.set(child.dataset.seq || '', child);
        });
        const keep = new Set();

        windowEntries.forEach((entry, index) => {
            const key = String(entry.seq);
            keep.add(key);
            let node = existing.get(key);
            if (!node) {
                node = createStreamNode(entry);
            } else {
                updateStreamNode(node, entry);
            }
            const currentChild = list.children[index];
            if (currentChild !== node) {
                list.insertBefore(node, currentChild || null);
            }
        });

        Array.from(list.children).forEach((child) => {
            if (!keep.has(child.dataset.seq || '')) child.remove();
        });

        refreshStreamChrome(visible.length);
        if (follow) scrollStreamToBottom();
        else restoreStreamAnchor(anchor);
    }

    function appendStreamNode(entry) {
        const list = els.piStreamList;
        if (!list) return false;
        const emptyNode = list.querySelector('.hud-empty');
        if (emptyNode) emptyNode.remove();

        const anchor = captureStreamAnchor();
        const node = createStreamNode(entry);
        node.classList.add('hud-stream-entry--enter');
        window.setTimeout(() => node.classList.remove('hud-stream-entry--enter'), 400);
        list.appendChild(node);

        while (list.children.length > streamRenderLimit) {
            const first = list.firstElementChild;
            if (!first || first === node) break;
            first.remove();
        }
        restoreStreamAnchor(anchor);
        return true;
    }

    function flushStreamUpdates() {
        streamFlushPending = false;
        let touched = false;
        streamDirtySeqs.forEach((seq) => {
            const entry = streamBySeq.get(seq);
            const node = streamNodeFor(seq);
            if (!entry || !node) return;
            updateStreamNode(node, entry);
            touched = true;
        });
        streamDirtySeqs.clear();
        if (touched && autoScrollState.stream) scrollStreamToBottom();
    }

    function scheduleStreamFlush() {
        if (streamFlushPending) return;
        streamFlushPending = true;
        requestAnimationFrame(flushStreamUpdates);
    }

    function onStreamEntry(raw) {
        const result = ingestStreamEntry(raw);
        if (!result) return;
        const { entry, isNew } = result;
        const seq = Number(entry.seq);
        const matches = buildStreamPredicate()(entry);

        if (!isNew) {
            const node = streamNodeFor(seq);
            if (node) {
                if (matches) {
                    streamDirtySeqs.add(seq);
                    scheduleStreamFlush();
                } else {
                    node.remove();
                }
            } else if (matches) {
                scheduleStreamRender();
            }
            scheduleStreamChrome();
            return;
        }

        if (!matches) {
            scheduleStreamChrome();
            return;
        }
        if (!autoScrollState.stream) {
            streamPendingNew += 1;
            updateStreamJump();
        }
        if (appendStreamNode(entry)) {
            if (autoScrollState.stream) scrollStreamToBottom();
            scheduleStreamChrome();
        } else {
            scheduleStreamRender();
        }
    }

    function jumpStreamToLive() {
        streamPendingNew = 0;
        autoScrollState.stream = true;
        if (streamRenderLimit !== STREAM_DOM_WINDOW) {
            streamRenderLimit = STREAM_DOM_WINDOW;
            renderStream();
            return;
        }
        scrollStreamToBottom();
        updateStreamJump();
    }

    function expandStreamWindow() {
        if (streamExpanding || !els.piStream) return;
        if (!els.piStreamOlder || els.piStreamOlder.hidden) return;
        streamExpanding = true;
        streamRenderLimit += STREAM_WINDOW_STEP;
        renderStream();
        streamExpanding = false;
    }

    function onStreamScroll() {
        const viewport = els.piStream;
        if (!viewport || streamExpanding) return;
        const wasFollowing = autoScrollState.stream;
        syncAutoScrollState('stream', viewport);
        if (autoScrollState.stream) {
            streamPendingNew = 0;
            if (!wasFollowing && streamRenderLimit !== STREAM_DOM_WINDOW) {
                streamRenderLimit = STREAM_DOM_WINDOW;
                scheduleStreamRender();
            }
        }
        updateStreamJump();
    }

    function initStreamControls() {
        const viewport = els.piStream;
        if (viewport) {
            syncAutoScrollState('stream', viewport);
            viewport.addEventListener('scroll', onStreamScroll, { passive: true });
        }
        if (els.piStreamJump) {
            els.piStreamJump.addEventListener('click', jumpStreamToLive);
        }
        if (els.piStreamOlder) {
            els.piStreamOlder.addEventListener('click', expandStreamWindow);
            if (viewport && typeof IntersectionObserver === 'function') {
                const observer = new IntersectionObserver(
                    (records) => {
                        if (records.some((record) => record.isIntersecting)) expandStreamWindow();
                    },
                    { root: viewport, rootMargin: '150px 0px 0px 0px' }
                );
                observer.observe(els.piStreamOlder);
            }
        }
        if (els.piStreamSearch) {
            let debounce = null;
            els.piStreamSearch.addEventListener('input', () => {
                clearTimeout(debounce);
                debounce = setTimeout(() => {
                    streamSearchQuery = els.piStreamSearch.value;
                    streamRenderLimit = STREAM_DOM_WINDOW;
                    renderStream();
                }, 180);
            });
        }
        renderStreamFilters();
    }

    async function fetchStreamPage(after, limit) {
        const params = new URLSearchParams();
        if (Number.isFinite(after) && after >= 0) params.set('after', String(after));
        params.set('limit', String(limit));
        const response = await fetch(api(`/supervisor/stream?${params.toString()}`));
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
    }

    function syncStream({ full = false } = {}) {
        if (streamSyncInFlight) return streamSyncInFlight;
        streamSyncInFlight = (async () => {
            let pendingReset = full;
            let after = full ? NaN : streamNextSeq - 1;
            for (let page = 0; page < STREAM_FETCH_PAGES; page += 1) {
                const payload = await fetchStreamPage(after, STREAM_FETCH_LIMIT);
                const sessionId = payload?.session_id || null;
                const sessionChanged = Boolean(
                    sessionId && streamSessionId && sessionId !== streamSessionId
                );
                if (pendingReset || sessionChanged) {
                    resetStream();
                    pendingReset = false;
                    after = NaN;
                }
                streamSessionId = sessionId || streamSessionId;
                const entries = Array.isArray(payload?.entries) ? payload.entries : [];
                entries.forEach((entry) => ingestStreamEntry(entry));
                if (Number.isFinite(payload?.next_seq)) {
                    streamNextSeq = Math.max(streamNextSeq, Number(payload.next_seq));
                }
                streamUnavailable = false;
                if (entries.length < STREAM_FETCH_LIMIT) break;
                after = streamNextSeq - 1;
            }
            renderStream();
        })()
            .catch(() => {
                streamUnavailable = true;
                if (!streamEntries.length) renderStream();
            })
            .finally(() => {
                streamSyncInFlight = null;
            });
        return streamSyncInFlight;
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

        applySteerAvailability(supervisor);

        els.piStderr.textContent = (supervisor.stderr_tail || []).join('\n') || 'No stderr output.';
        els.rawSupervisor.textContent = formatJSON(supervisor);
        renderCritique(supervisor);
    }

    function critiqueStateLabel(critique) {
        if (!critique.enabled) return 'DISABLED';
        if (critique.salvaged) return 'SALVAGED';
        if (critique.text) return 'READY';
        if (critique.error) return 'FAILED';
        return 'NONE';
    }

    // The whole retrospective, scrollable and never elided: this text is the most
    // valuable thing the run produces and the operator reads it end to end.
    function renderCritique(supervisor) {
        const critique = supervisor.critique || {};
        const state = critiqueStateLabel(critique);
        els.critiqueStateChip.textContent = state;
        els.critiqueStateChip.dataset.status = state.toLowerCase();

        const meta = [];
        if (critique.at) meta.push(timeLabel(critique.at));
        if (critique.duration_seconds) meta.push(`${Math.round(critique.duration_seconds)}s`);
        if (critique.digest_tokens) meta.push(`${formatCompactNumber(critique.digest_tokens)} tok`);
        els.critiqueMetaChip.textContent = meta.join(' · ') || '—';

        const source = String(supervisor.goal_source || '').toUpperCase();
        els.critiqueActiveGoal.textContent = supervisor.goal || 'No goal set.';
        els.critiqueGoalSource.textContent = `SOURCE: ${source || '—'}`;
        els.critiqueNextGoal.textContent =
            critique.next_goal || supervisor.critic_next_goal || 'None proposed yet.';

        els.critiqueError.textContent = critique.error ? `► ${critique.error}` : '';
        els.critiqueError.hidden = !critique.error;
        els.critiqueText.textContent = critique.text || 'No retrospective yet.';
    }

    /* ------------------------------------------------------------------ */
    /* Campaign: progress rail, presses to milestone, run health           */
    /* ------------------------------------------------------------------ */

    const BENCHMARK_ROWS = [
        { key: 'ours-presses', group: 'Presses', role: 'ours', name: 'This run' },
        { key: 'pa-best',      group: 'Presses', role: 'ref',  name: 'PokeAgent best' },
        { key: 'pa-eff',       group: 'Presses', role: 'ref',  name: 'PokeAgent efficient' },
        { key: 'ours-clock',   group: 'Clock',   role: 'ours', name: 'This run' },
        { key: 'human',        group: 'Clock',   role: 'ref',  name: 'Human speedrun' },
    ];

    const progressState = {
        available: false,
        lastAttempt: 0,
        count: null,
        total: RED_LADDER.length,
        furthest: null,
        furthestLabel: '',
        latest: [],
        presses: null,
        startedAt: null,
        elapsedSeconds: null,
        runKey: '',
        health: {},
    };

    const pressLedger = new Map();   // ladder id -> { presses, at, seconds, source }
    const pressSamples = [];         // { t, presses } — the window the rate is read from
    let ledgerBaseline = false;

    const healthState = {
        lastObservationId: null,
        observations: 0,
        blocked: 0,
        positionSamples: 0,
        uniquePositions: new Set(),
        lastPartyHp: null,
        whiteouts: 0,
        reloads: 0,
        historyWhiteouts: 0,
        historyReloads: 0,
    };

    let railButtons = [];
    let ladderRowCells = [];
    let benchmarkRows = new Map();
    let healthPills = new Map();
    let railFocusIndex = 0;
    let railPinnedIndex = 0;
    // What the pointer or keyboard is on right now, so a 3s poll does not yank
    // the readout out from under someone reading it.
    let railHoverIndex = null;

    // Number(null) and Number('') are both 0, which would silently invent a
    // reading of zero out of a field the server left empty. Everything that
    // reads a number off /progress goes through here.
    function finiteOrNull(value) {
        if (value === null || value === undefined || value === '') return null;
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }

    function formatInt(value) {
        const number = Number(value);
        if (!Number.isFinite(number)) return '—';
        return Math.round(number).toLocaleString('en-US');
    }

    function formatRate(ratio) {
        if (!Number.isFinite(ratio)) return '—';
        return `${(ratio * 100).toFixed(ratio < 0.1 ? 1 : 0)}%`;
    }

    function formatDuration(seconds) {
        if (!Number.isFinite(seconds) || seconds < 0) return '—';
        const total = Math.round(seconds);
        const hours = Math.floor(total / 3600);
        const minutes = Math.floor((total % 3600) / 60);
        const rest = total % 60;
        if (hours) return `${hours}h ${String(minutes).padStart(2, '0')}m`;
        if (minutes) return `${minutes}m ${String(rest).padStart(2, '0')}s`;
        return `${rest}s`;
    }

    function resolveRung(reference) {
        if (!reference) return null;
        if (typeof reference === 'object') {
            return resolveRung(
                reference.id || reference.milestone_id || reference.milestone || reference.label
            );
        }
        const key = String(reference).trim();
        if (!key) return null;
        return LADDER_BY_ID.get(key) || LADDER_BY_LABEL.get(key.toLowerCase()) || null;
    }

    /* -- press ledger, kept across reloads so the curve survives F5 ------ */

    function loadLedger(runKey) {
        pressLedger.clear();
        try {
            const raw = window.localStorage.getItem(LEDGER_STORAGE_KEY);
            if (!raw) return;
            const saved = JSON.parse(raw);
            if (!saved || saved.runKey !== runKey || !Array.isArray(saved.entries)) return;
            saved.entries.forEach((pair) => {
                if (!Array.isArray(pair)) return;
                const [id, value] = pair;
                if (!LADDER_BY_ID.has(id) || !value) return;
                const presses = finiteOrNull(value.presses);
                if (presses === null) return;
                pressLedger.set(id, {
                    presses,
                    at: finiteOrNull(value.at),
                    seconds: finiteOrNull(value.seconds),
                    source: value.source === 'server' ? 'server' : 'observed',
                });
            });
        } catch (error) {
            // Private mode, blocked storage, corrupt value: start from an empty ledger.
        }
    }

    function saveLedger(runKey) {
        try {
            window.localStorage.setItem(
                LEDGER_STORAGE_KEY,
                JSON.stringify({ runKey, entries: Array.from(pressLedger.entries()) })
            );
        } catch (error) {
            // Storage unavailable: the in-memory ledger still drives this session.
        }
    }

    function ingestServerLedger(data) {
        let changed = false;
        const record = (reference, presses, seconds) => {
            const rung = resolveRung(reference);
            const cost = finiteOrNull(presses);
            if (!rung || cost === null) return;
            const existing = pressLedger.get(rung.id);
            if (existing && existing.source === 'server' && existing.presses === cost) return;
            const clock = finiteOrNull(seconds);
            pressLedger.set(rung.id, {
                presses: cost,
                at: existing ? existing.at : null,
                seconds: clock !== null ? clock : (existing ? existing.seconds : null),
                source: 'server',
            });
            changed = true;
        };

        const pressesTo = data.presses_to;
        if (pressesTo && typeof pressesTo === 'object' && !Array.isArray(pressesTo)) {
            Object.keys(pressesTo).forEach((id) => record(id, pressesTo[id], null));
        }
        if (Array.isArray(data.attainments)) {
            data.attainments.forEach((entry) => record(entry, entry && entry.presses, entry && entry.seconds));
        }
        return changed;
    }

    function progressRunKey(data) {
        const explicit = data.run_id || data.run || data.session_id;
        if (explicit) return String(explicit);
        if (data.started_at) return `start:${data.started_at}`;
        return 'unkeyed';
    }

    function applyProgressPayload(payload) {
        const data = (payload && typeof payload === 'object') ? payload : {};
        const runKey = progressRunKey(data);
        const count = finiteOrNull(data.count);
        const presses = finiteOrNull(data.presses);
        const previousCount = progressState.count;
        const previousPresses = progressState.presses;

        // A counter that went backwards means a different run, not a fixed one.
        const rewound =
            (count !== null && previousCount !== null && count < previousCount) ||
            (presses !== null && previousPresses !== null && presses < previousPresses);

        if (runKey !== progressState.runKey || rewound) {
            progressState.runKey = runKey;
            pressSamples.length = 0;
            ledgerBaseline = false;
            if (rewound) {
                pressLedger.clear();
                saveLedger(runKey);
            } else {
                loadLedger(runKey);
            }
        }

        progressState.available = true;
        progressState.count = count;
        progressState.total = (finiteOrNull(data.total) || 0) > 0
            ? finiteOrNull(data.total)
            : RED_LADDER.length;
        progressState.presses = presses;
        progressState.latest = Array.isArray(data.latest) ? data.latest.slice() : [];
        progressState.startedAt = data.started_at || null;
        progressState.elapsedSeconds = finiteOrNull(data.elapsed_seconds);
        progressState.health = (data.health && typeof data.health === 'object') ? data.health : data;

        const furthestRung = resolveRung(data.furthest) || resolveRung(data.furthest_label);
        progressState.furthest = furthestRung
            ? furthestRung.id
            : (data.furthest ? String(data.furthest) : null);
        progressState.furthestLabel =
            data.furthest_label || (furthestRung ? furthestRung.label : '');

        // A server that prices the rungs itself always outranks what this page saw.
        let ledgerChanged = ingestServerLedger(data);

        if (presses !== null) {
            pressSamples.push({ t: Date.now(), presses });
            while (pressSamples.length > PRESS_SAMPLE_LIMIT) pressSamples.shift();

            // Only price a rung this page watched appear. On the first payload the
            // ladder is already partly climbed and today's press count says nothing
            // about what those earlier rungs cost.
            const climbed =
                ledgerBaseline && count !== null && previousCount !== null && count > previousCount;
            if (climbed) {
                const seen = progressState.latest.map(resolveRung).filter(Boolean);
                if (furthestRung) seen.push(furthestRung);
                seen.forEach((rung) => {
                    if (pressLedger.has(rung.id)) return;
                    pressLedger.set(rung.id, {
                        presses,
                        at: Date.now(),
                        seconds: null,
                        source: 'observed',
                    });
                    ledgerChanged = true;
                });
            }
            ledgerBaseline = true;
        }

        if (ledgerChanged) saveLedger(progressState.runKey);
    }

    /* -- derived readings ----------------------------------------------- */

    function furthestIndex() {
        const rung = progressState.furthest ? LADDER_BY_ID.get(progressState.furthest) : null;
        if (rung) return rung.index;
        if (Number.isFinite(progressState.count) && progressState.count > 0) {
            return Math.min(RED_LADDER.length - 1, progressState.count - 1);
        }
        return -1;
    }

    function rungState(index) {
        const furthest = furthestIndex();
        if (index <= furthest) return 'done';
        if (index === furthest + 1 && progressState.available) return 'current';
        return 'todo';
    }

    function rungsReached() {
        if (Number.isFinite(progressState.count)) return progressState.count;
        return Math.max(0, furthestIndex() + 1);
    }

    function ledgerPoints() {
        const points = [];
        pressLedger.forEach((value, id) => {
            const rung = LADDER_BY_ID.get(id);
            if (!rung || !Number.isFinite(value.presses)) return;
            points.push({
                index: rung.index,
                id,
                label: rung.label,
                presses: value.presses,
                at: value.at,
                source: value.source,
            });
        });
        points.sort((a, b) => a.index - b.index);
        return points;
    }

    function pressRatePerMinute() {
        if (pressSamples.length < 2) return null;
        const first = pressSamples[0];
        const last = pressSamples[pressSamples.length - 1];
        const minutes = (last.t - first.t) / 60000;
        if (minutes < 0.25) return null;
        const delta = last.presses - first.presses;
        if (delta < 0) return null;
        return delta / minutes;
    }

    function runStartMs() {
        const started = parseTs(progressState.startedAt);
        return Number.isNaN(started) ? null : started;
    }

    function runElapsed() {
        if (Number.isFinite(progressState.elapsedSeconds)) {
            return { seconds: progressState.elapsedSeconds, exact: true };
        }
        const start = runStartMs();
        if (start !== null) return { seconds: (Date.now() - start) / 1000, exact: true };
        if (pressSamples.length) {
            return { seconds: (Date.now() - pressSamples[0].t) / 1000, exact: false };
        }
        return { seconds: null, exact: false };
    }

    function firstGymSeconds() {
        const gym = pressLedger.get(FIRST_GYM_ID);
        if (!gym) return null;
        if (Number.isFinite(gym.seconds)) return gym.seconds;
        const start = runStartMs();
        if (start !== null && Number.isFinite(gym.at)) return Math.max(0, (gym.at - start) / 1000);
        return null;
    }

    function severityFor(value, warnAt, critAt) {
        if (!Number.isFinite(value)) return 'idle';
        if (value >= critAt) return 'crit';
        if (value >= warnAt) return 'warn';
        return 'good';
    }

    /* -- run health ------------------------------------------------------ */

    function toolCallHealth() {
        let calls = 0;
        let errors = 0;
        streamEntries.forEach((entry) => {
            if (streamEntryKind(entry) !== 'tool') return;
            const state = streamEntryState(entry);
            if (state === 'running') return;
            calls += 1;
            if (state === 'error') errors += 1;
        });
        return { calls, errors };
    }

    function recordObservationSample(payload) {
        const stamp = payload.observation_id || payload.generated_at;
        if (!stamp || stamp === healthState.lastObservationId) return;
        healthState.lastObservationId = stamp;
        healthState.observations += 1;

        const world = payload.world_state || {};
        const interaction = world.interaction || {};
        if (String(interaction.source || '') === 'blocked_tile') {
            healthState.blocked += 1;
        }

        const position = (world.player || {}).position || {};
        const x = Number(position.x);
        const y = Number(position.y);
        if (Number.isFinite(x) && Number.isFinite(y)) {
            healthState.positionSamples += 1;
            healthState.uniquePositions.add(`${(world.map || {}).map_name || '?'}|${x},${y}`);
        }

        const party = Array.isArray(world.party) ? world.party : [];
        if (party.length) {
            const hp = party.reduce((total, mon) => total + (Number(mon.hp) || 0), 0);
            if (healthState.lastPartyHp !== null && healthState.lastPartyHp > 0 && hp === 0) {
                healthState.whiteouts += 1;
            }
            healthState.lastPartyHp = hp;
        }
    }

    function recordHistoryHealth(events) {
        let whiteouts = 0;
        let reloads = 0;
        events.forEach((event) => {
            const type = String(event.type || '').toLowerCase();
            let blob = type;
            try {
                blob = `${type} ${JSON.stringify(event)}`.toLowerCase();
            } catch (error) {
                // Cyclic or exotic payload: the type alone still classifies it.
            }
            if (/white[_\s-]?out|black[_\s-]?out/.test(blob)) whiteouts += 1;
            if (type === 'load' || type === 'recovery' || /\bload\b|restore/.test(type)) reloads += 1;
        });
        healthState.historyWhiteouts = whiteouts;
        healthState.historyReloads = reloads;
    }

    function healthReadings() {
        const server = progressState.health || {};
        const tools = toolCallHealth();
        const num = finiteOrNull;
        const pick = (serverValue, local) => (serverValue !== null ? serverValue : local);
        const note = (serverValue, localNote) => (serverValue !== null ? 'scored by server' : localNote);

        const blockedServer = num(server.blocked_rate);
        const toolServer = num(server.tool_error_rate);
        const revisitServer = num(server.revisit_ratio);
        const whiteoutServer = num(server.whiteouts);
        const reloadServer = num(server.reloads);

        const blocked = pick(
            blockedServer,
            healthState.observations ? healthState.blocked / healthState.observations : null
        );
        const toolErr = pick(toolServer, tools.calls ? tools.errors / tools.calls : null);
        const revisit = pick(
            revisitServer,
            healthState.uniquePositions.size
                ? healthState.positionSamples / healthState.uniquePositions.size
                : null
        );
        const whiteouts = pick(
            whiteoutServer,
            Math.max(healthState.whiteouts, healthState.historyWhiteouts)
        );
        const reloads = pick(reloadServer, Math.max(healthState.reloads, healthState.historyReloads));

        return {
            blocked: {
                value: blocked,
                text: formatRate(blocked),
                note: note(blockedServer, `${formatInt(healthState.blocked)} of ${formatInt(healthState.observations)} obs`),
            },
            toolerr: {
                value: toolErr,
                text: formatRate(toolErr),
                note: note(toolServer, `${formatInt(tools.errors)} of ${formatInt(tools.calls)} calls`),
            },
            revisit: {
                value: revisit,
                text: Number.isFinite(revisit) ? `${revisit.toFixed(2)}×` : '—',
                note: note(
                    revisitServer,
                    `${formatInt(healthState.positionSamples)} steps / ${formatInt(healthState.uniquePositions.size)} tiles`
                ),
            },
            whiteouts: {
                value: whiteouts,
                text: formatInt(whiteouts),
                note: note(whiteoutServer, 'party wiped'),
            },
            reloads: {
                value: reloads,
                text: formatInt(reloads),
                note: note(reloadServer, 'save restored'),
            },
        };
    }

    function thresholdText(spec) {
        const asText = (value) => (spec.crit < 1 ? `${(value * 100).toFixed(0)}%` : String(value));
        return `warning at ${asText(spec.warn)}, critical at ${asText(spec.crit)}`;
    }

    /* -- build (once) ---------------------------------------------------- */

    function buildRail() {
        if (!els.campaignRail) return;
        const fragment = document.createDocumentFragment();
        railButtons = RED_LADDER.map((rung) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'hud-rung';
            button.dataset.index = String(rung.index);
            button.dataset.kind = rung.kind;
            button.dataset.state = 'todo';
            button.tabIndex = -1;
            fragment.appendChild(button);
            return button;
        });
        els.campaignRail.replaceChildren(fragment);
        if (railButtons.length) railButtons[0].tabIndex = 0;
        els.campaignRail.addEventListener('mouseover', onRailPoint);
        els.campaignRail.addEventListener('focusin', onRailPoint);
        els.campaignRail.addEventListener('mouseleave', onRailLeave);
        els.campaignRail.addEventListener('focusout', onRailLeave);
        els.campaignRail.addEventListener('click', onRailClick);
        els.campaignRail.addEventListener('keydown', onRailKeydown);
    }

    function rungFromEvent(event) {
        const target = event.target;
        if (!target || typeof target.closest !== 'function') return null;
        const button = target.closest('.hud-rung');
        if (!button) return null;
        const index = Number(button.dataset.index);
        return Number.isFinite(index) ? index : null;
    }

    function onRailPoint(event) {
        const index = rungFromEvent(event);
        if (index === null) return;
        railHoverIndex = index;
        setRailReadout(index);
    }

    function onRailLeave() {
        railHoverIndex = null;
        setRailReadout(null);
    }

    function moveRailFocus(index) {
        if (!railButtons.length) return;
        const clamped = Math.max(0, Math.min(railButtons.length - 1, index));
        railButtons.forEach((button, position) => {
            button.tabIndex = position === clamped ? 0 : -1;
        });
        railFocusIndex = clamped;
    }

    function onRailClick(event) {
        const index = rungFromEvent(event);
        if (index === null) return;
        moveRailFocus(index);
        if (!els.campaignLadderDetails) return;
        els.campaignLadderDetails.open = true;
        const cells = ladderRowCells[index];
        if (cells && cells.row && typeof cells.row.scrollIntoView === 'function') {
            cells.row.scrollIntoView({ block: 'nearest' });
        }
    }

    function onRailKeydown(event) {
        const steps = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -1, ArrowDown: 1 };
        let next = null;
        if (Object.prototype.hasOwnProperty.call(steps, event.key)) {
            next = railFocusIndex + steps[event.key];
        } else if (event.key === 'Home') {
            next = 0;
        } else if (event.key === 'End') {
            next = railButtons.length - 1;
        } else {
            return;
        }
        event.preventDefault();
        moveRailFocus(next);
        const button = railButtons[railFocusIndex];
        if (button) button.focus();
    }

    function buildLadderRows() {
        if (!els.campaignLadderRows) return;
        const fragment = document.createDocumentFragment();
        ladderRowCells = RED_LADDER.map((rung) => {
            const row = document.createElement('tr');
            row.dataset.index = String(rung.index);
            row.dataset.kind = rung.kind;
            row.dataset.state = 'todo';

            const ordinal = document.createElement('td');
            ordinal.textContent = String(rung.index + 1);
            const label = document.createElement('td');
            label.textContent = rung.label;
            label.title = rung.id;
            const kind = document.createElement('td');
            kind.className = 'hud-ladder-kind';
            kind.textContent = rung.kind;
            const presses = document.createElement('td');
            presses.textContent = '—';
            const delta = document.createElement('td');
            delta.textContent = '';

            row.append(ordinal, label, kind, presses, delta);
            fragment.appendChild(row);
            return { row, presses, delta };
        });
        els.campaignLadderRows.replaceChildren(fragment);
    }

    function buildBenchmark() {
        if (!els.campaignBenchmark) return;
        const fragment = document.createDocumentFragment();
        benchmarkRows = new Map();
        let currentGroup = null;
        let groupNode = null;
        BENCHMARK_ROWS.forEach((spec) => {
            if (spec.group !== currentGroup) {
                currentGroup = spec.group;
                groupNode = document.createElement('div');
                groupNode.className = 'hud-benchmark-group';
                const kicker = document.createElement('span');
                kicker.className = 'hud-kicker-mini';
                kicker.textContent = spec.group;
                groupNode.appendChild(kicker);
                fragment.appendChild(groupNode);
            }
            const row = document.createElement('div');
            row.className = 'hud-benchmark-row';
            row.dataset.role = spec.role;
            row.dataset.sev = 'idle';
            const name = document.createElement('span');
            name.className = 'hud-benchmark-name';
            name.textContent = spec.name;
            const value = document.createElement('span');
            value.className = 'hud-benchmark-value';
            value.textContent = '—';
            const bar = document.createElement('span');
            bar.className = 'hud-benchmark-bar';
            const fill = document.createElement('span');
            fill.className = 'hud-benchmark-bar-fill';
            bar.appendChild(fill);
            row.append(name, bar, value);
            groupNode.appendChild(row);
            benchmarkRows.set(spec.key, { row, value, fill });
        });
        els.campaignBenchmark.replaceChildren(fragment);
    }

    function buildHealthStrip() {
        if (!els.healthStrip) return;
        const fragment = document.createDocumentFragment();
        healthPills = new Map();
        HEALTH_SPECS.forEach((spec) => {
            const pill = document.createElement('div');
            pill.className = 'hud-health-pill';
            pill.dataset.key = spec.key;
            pill.dataset.sev = 'idle';

            const glyph = document.createElement('span');
            glyph.className = 'hud-health-glyph';
            glyph.setAttribute('aria-hidden', 'true');
            glyph.textContent = HEALTH_GLYPH.idle;
            const label = document.createElement('span');
            label.className = 'hud-health-label';
            label.textContent = spec.label;
            const value = document.createElement('strong');
            value.className = 'hud-health-value';
            value.textContent = '—';
            const meter = document.createElement('span');
            meter.className = 'hud-health-meter';
            const fill = document.createElement('span');
            fill.className = 'hud-health-meter-fill';
            meter.appendChild(fill);
            const note = document.createElement('span');
            note.className = 'hud-health-note';
            note.textContent = 'no samples yet';

            pill.append(glyph, label, value, meter, note);
            fragment.appendChild(pill);
            healthPills.set(spec.key, { pill, glyph, value, fill, note });
        });
        els.healthStrip.replaceChildren(fragment);
    }

    /* -- render (every refresh) ------------------------------------------ */

    function setRailReadout(index) {
        if (!els.campaignRailReadout) return;
        const position = (index === null || index === undefined) ? railPinnedIndex : index;
        const rung = RED_LADDER[position];
        if (!rung) return;
        const state = rungState(rung.index);
        const word = state === 'done' ? 'reached' : (state === 'current' ? 'next up' : 'ahead');
        const priced = pressLedger.get(rung.id);
        const price = priced
            ? `${formatInt(priced.presses)} presses`
            : (state === 'done' ? 'price not recorded' : 'unpriced');
        els.campaignRailReadout.innerHTML =
            `<strong>${String(rung.index + 1).padStart(2, '0')} · ${escapeHtml(rung.label)}</strong> ` +
            `<span class="hud-dim">${escapeHtml(rung.kind)}</span> — ${word} · ` +
            `<span class="hud-rail-price">${escapeHtml(price)}</span>`;
    }

    function renderCampaignChips() {
        const total = progressState.total || RED_LADDER.length;
        if (els.campaignRungChip) {
            els.campaignRungChip.textContent = `RUNG ${formatInt(rungsReached())}/${formatInt(total)}`;
        }
        if (els.campaignPressesChip) {
            els.campaignPressesChip.textContent = `PRESSES ${formatInt(progressState.presses)}`;
        }
        if (els.campaignSourceChip) {
            els.campaignSourceChip.textContent = progressState.available
                ? 'SOURCE: /PROGRESS'
                : 'SOURCE: OFFLINE';
            els.campaignSourceChip.dataset.status = progressState.available ? 'running' : 'stopping';
        }
        if (!els.campaignHeadline) return;
        if (!progressState.available) {
            els.campaignHeadline.textContent =
                'Scoreboard offline — /progress is not answering. The full 63-rung ladder is below.';
            return;
        }
        const furthest = furthestIndex();
        const next = RED_LADDER[Math.min(RED_LADDER.length - 1, furthest + 1)];
        const nextText = furthest >= RED_LADDER.length - 1
            ? 'ladder complete'
            : `next: ${escapeHtml(next ? next.label : '—')}`;
        if (furthest < 0) {
            els.campaignHeadline.innerHTML =
                `<em>No rung reached yet</em> · <b>${formatInt(progressState.presses)}</b> presses · ${nextText}`;
            return;
        }
        const name = progressState.furthestLabel || RED_LADDER[furthest].label;
        // The rail is positional; `count` is what the server actually confirmed.
        // When they disagree the run passed a rung whose flag never read true,
        // and the operator should see that rather than guess which number wins.
        const gap = Number.isFinite(progressState.count)
            ? (furthest + 1) - progressState.count
            : 0;
        const unconfirmed = gap > 0
            ? ` · <span class="hud-dim">${gap} unconfirmed</span>`
            : '';
        els.campaignHeadline.innerHTML =
            `<em>${escapeHtml(name)}</em> · rung ${furthest + 1} of ${total} · ` +
            `<b>${formatInt(progressState.presses)}</b> presses · ${nextText}${unconfirmed}`;
    }

    function renderCampaignStats() {
        if (!els.campaignStats) return;
        const reached = rungsReached();
        const presses = progressState.presses;
        const perRung = (Number.isFinite(presses) && reached > 0) ? presses / reached : null;
        const points = ledgerPoints();
        const lastPriced = points.length ? points[points.length - 1] : null;
        const sinceLast = (Number.isFinite(presses) && lastPriced)
            ? Math.max(0, presses - lastPriced.presses)
            : null;
        const rate = pressRatePerMinute();
        const elapsed = runElapsed();
        renderKeyValueCards(els.campaignStats, [
            ['RUNGS', `${formatInt(reached)}/${formatInt(progressState.total || RED_LADDER.length)}`, 'ladder'],
            ['PRESSES', formatInt(presses), 'presses'],
            ['PRESSES/RUNG', perRung === null ? '—' : formatInt(perRung), 'presses'],
            ['SINCE LAST', sinceLast === null ? '—' : formatInt(sinceLast), 'presses'],
            ['RATE', rate === null ? '—' : `${rate.toFixed(0)}/min`, 'clock'],
            [
                'ELAPSED',
                elapsed.seconds === null
                    ? '—'
                    : `${elapsed.exact ? '' : '~'}${formatDuration(elapsed.seconds)}`,
                'clock',
            ],
        ]);
    }

    function renderRail() {
        if (!railButtons.length) return;
        const furthest = furthestIndex();
        railButtons.forEach((button, index) => {
            const rung = RED_LADDER[index];
            const state = rungState(index);
            if (button.dataset.state !== state) button.dataset.state = state;
            const priced = pressLedger.get(rung.id);
            const label = `${index + 1}. ${rung.label} — ${state}` +
                (priced ? ` at ${formatInt(priced.presses)} presses` : '');
            if (button.title !== label) {
                button.title = label;
                button.setAttribute('aria-label', label);
            }
        });
        const fraction = Math.max(0, Math.min(1, (furthest + 1) / RED_LADDER.length));
        if (els.campaignRailFill) {
            els.campaignRailFill.style.width = `${(fraction * 100).toFixed(2)}%`;
        }
        railPinnedIndex = Math.max(0, Math.min(RED_LADDER.length - 1, furthest + 1));
    }

    function svgNode(name, attrs) {
        const node = document.createElementNS('http://www.w3.org/2000/svg', name);
        Object.keys(attrs || {}).forEach((key) => node.setAttribute(key, String(attrs[key])));
        return node;
    }

    function renderPressChart() {
        const svg = els.campaignChart;
        if (!svg) return;
        svg.replaceChildren();

        const WIDTH = 320;
        const HEIGHT = 96;   // must match the viewBox on #campaignChart
        const padLeft = 38;
        const padRight = 10;
        const padTop = 8;
        const padBottom = 15;
        const plotW = WIDTH - padLeft - padRight;
        const plotH = HEIGHT - padTop - padBottom;
        const rungs = RED_LADDER.length;

        const series = ledgerPoints().map((point) => ({ x: point.index, y: point.presses }));
        const furthest = furthestIndex();
        const presses = progressState.presses;
        if (Number.isFinite(presses)) {
            const liveX = Math.max(0, Math.min(rungs - 1, furthest + 1));
            const last = series[series.length - 1];
            if (!last || (presses >= last.y && liveX >= last.x)) {
                series.push({ x: liveX, y: presses, live: true });
            }
        }

        if (!series.length) {
            if (els.campaignChartCaption) {
                els.campaignChartCaption.textContent =
                    'No press ledger yet — rungs get priced as the run reaches them.';
            }
            return;
        }

        const gymRung = LADDER_BY_ID.get(FIRST_GYM_ID);
        let yMax = series.reduce((top, point) => Math.max(top, point.y), 0);
        if (gymRung && yMax < REF_POKEAGENT_BEST) yMax = REF_POKEAGENT_BEST;
        yMax = Math.max(1, yMax);

        const xOf = (index) => padLeft + (rungs <= 1 ? 0 : (index / (rungs - 1)) * plotW);
        const yOf = (value) => padTop + plotH - (Math.max(0, Math.min(value, yMax)) / yMax) * plotH;

        // Faint grid first, so the data always sits on top of it.
        [0, 0.5, 1].forEach((fraction) => {
            const y = yOf(yMax * fraction);
            svg.appendChild(svgNode('line', {
                x1: padLeft, x2: WIDTH - padRight, y1: y, y2: y,
                stroke: 'var(--hud-grid)', 'stroke-width': 1,
            }));
            const label = svgNode('text', { x: padLeft - 4, y: y + 2.5, 'text-anchor': 'end' });
            label.textContent = fraction === 0 ? '0' : formatInt(yMax * fraction);
            svg.appendChild(label);
        });

        let line = `M ${xOf(series[0].x)} ${yOf(series[0].y)}`;
        for (let i = 1; i < series.length; i += 1) {
            line += ` L ${xOf(series[i].x)} ${yOf(series[i - 1].y)}`;
            line += ` L ${xOf(series[i].x)} ${yOf(series[i].y)}`;
        }

        const baseline = yOf(0);
        svg.appendChild(svgNode('path', {
            d: `${line} L ${xOf(series[series.length - 1].x)} ${baseline} L ${xOf(series[0].x)} ${baseline} Z`,
            fill: 'var(--hud-cyan-soft)',
            stroke: 'none',
        }));
        svg.appendChild(svgNode('path', {
            d: line,
            fill: 'none',
            stroke: 'var(--hud-cyan)',
            'stroke-width': 1.4,
            'stroke-linejoin': 'round',
        }));

        // The two published first-gym numbers, plotted where they belong: at the
        // gym rung, so the curve either clears them or does not.
        if (gymRung) {
            const gymX = xOf(gymRung.index);
            svg.appendChild(svgNode('line', {
                x1: gymX, x2: gymX, y1: padTop, y2: baseline,
                stroke: 'var(--hud-line)', 'stroke-width': 1, 'stroke-dasharray': '2 3',
            }));
            [
                [REF_POKEAGENT_EFFICIENT, 'eff 649'],
                [REF_POKEAGENT_BEST, 'best 1608'],
            ].forEach(([value, text]) => {
                if (value > yMax) return;
                const y = yOf(value);
                svg.appendChild(svgNode('circle', {
                    cx: gymX, cy: y, r: 2.4,
                    fill: 'none', stroke: 'var(--hud-ghost)', 'stroke-width': 1,
                }));
                const label = svgNode('text', { x: gymX + 5, y: y - 3, class: 'hud-chart-ref' });
                label.textContent = text;
                svg.appendChild(label);
            });
        }

        // Emphasised endpoint: where the run stands right now.
        const end = series[series.length - 1];
        svg.appendChild(svgNode('circle', {
            cx: xOf(end.x), cy: yOf(end.y), r: 5,
            fill: 'var(--hud-hazard)', opacity: 0.22,
        }));
        svg.appendChild(svgNode('circle', {
            cx: xOf(end.x), cy: yOf(end.y), r: 2.6, fill: 'var(--hud-hazard)',
        }));

        svg.appendChild(svgNode('line', {
            x1: padLeft, x2: WIDTH - padRight, y1: baseline, y2: baseline,
            stroke: 'var(--hud-line)', 'stroke-width': 1,
        }));
        const first = svgNode('text', { x: padLeft, y: HEIGHT - 6, class: 'hud-chart-axis-label' });
        first.textContent = 'rung 1';
        svg.appendChild(first);
        const last = svgNode('text', {
            x: WIDTH - padRight, y: HEIGHT - 6, 'text-anchor': 'end', class: 'hud-chart-axis-label',
        });
        last.textContent = `rung ${rungs}`;
        svg.appendChild(last);

        if (els.campaignChartCaption) {
            const priced = ledgerPoints().length;
            els.campaignChartCaption.textContent = priced
                ? `${formatInt(priced)} rungs priced · peak ${formatInt(yMax)} presses on the y-axis`
                : 'Live position only — no rung has been priced during this session yet.';
        }
    }

    function renderBenchmark() {
        if (!benchmarkRows.size) return;
        const gym = pressLedger.get(FIRST_GYM_ID);
        const ourPresses = gym && Number.isFinite(gym.presses) ? gym.presses : null;
        const ourClock = firstGymSeconds();

        const pressScale = Math.max(ourPresses || 0, REF_POKEAGENT_BEST);
        const clockScale = Math.max(ourClock || 0, REF_HUMAN_SPEEDRUN_SECONDS);

        const apply = (key, text, value, scale, sev) => {
            const node = benchmarkRows.get(key);
            if (!node) return;
            node.value.textContent = text;
            node.row.dataset.sev = sev || 'idle';
            const width = (Number.isFinite(value) && scale > 0)
                ? Math.max(0, Math.min(100, (value / scale) * 100))
                : 0;
            node.fill.style.width = `${width.toFixed(1)}%`;
        };

        apply(
            'ours-presses',
            ourPresses === null ? 'not reached' : `${formatInt(ourPresses)} presses`,
            ourPresses,
            pressScale,
            ourPresses === null
                ? 'idle'
                : severityFor(ourPresses, REF_POKEAGENT_EFFICIENT, REF_POKEAGENT_BEST)
        );
        apply('pa-best', `${formatInt(REF_POKEAGENT_BEST)} actions`, REF_POKEAGENT_BEST, pressScale, 'idle');
        apply('pa-eff', `${formatInt(REF_POKEAGENT_EFFICIENT)} actions`, REF_POKEAGENT_EFFICIENT, pressScale, 'idle');
        apply(
            'ours-clock',
            ourClock === null ? 'not observed' : formatDuration(ourClock),
            ourClock,
            clockScale,
            ourClock === null
                ? 'idle'
                : severityFor(
                    ourClock,
                    REF_HUMAN_SPEEDRUN_SECONDS * 3,
                    REF_HUMAN_SPEEDRUN_SECONDS * 10
                )
        );
        apply('human', '~18m', REF_HUMAN_SPEEDRUN_SECONDS, clockScale, 'idle');
    }

    function renderLadderRows() {
        if (!ladderRowCells.length) return;
        const priceByIndex = new Map(ledgerPoints().map((point) => [point.index, point.presses]));
        let previous = null;
        RED_LADDER.forEach((rung) => {
            const cells = ladderRowCells[rung.index];
            if (!cells) return;
            const state = rungState(rung.index);
            if (cells.row.dataset.state !== state) cells.row.dataset.state = state;
            const price = priceByIndex.get(rung.index);
            if (Number.isFinite(price)) {
                cells.presses.textContent = formatInt(price);
                cells.delta.textContent = previous === null ? '' : `+${formatInt(price - previous)}`;
                previous = price;
            } else {
                cells.presses.textContent = state === 'done' ? '·' : '—';
                cells.delta.textContent = '';
            }
        });
    }

    function renderHealth() {
        if (!healthPills.size) return;
        const readings = healthReadings();
        HEALTH_SPECS.forEach((spec) => {
            const node = healthPills.get(spec.key);
            const reading = readings[spec.key];
            if (!node || !reading) return;
            const level = severityFor(reading.value, spec.warn, spec.crit);
            node.pill.dataset.sev = level;
            node.pill.title = `${spec.label}: ${reading.text} — ${thresholdText(spec)}`;
            node.glyph.textContent = HEALTH_GLYPH[level];
            node.value.textContent = reading.text;
            node.note.textContent = reading.note;
            const width = (Number.isFinite(reading.value) && spec.crit > 0)
                ? Math.max(0, Math.min(100, (reading.value / spec.crit) * 100))
                : 0;
            node.fill.style.width = `${width.toFixed(1)}%`;
        });
        if (els.healthWindowChip) {
            const tools = toolCallHealth();
            els.healthWindowChip.textContent =
                `WINDOW: ${formatInt(healthState.observations)} OBS · ${formatInt(tools.calls)} TOOL`;
        }
    }

    function renderCampaign() {
        renderCampaignChips();
        renderCampaignStats();
        renderRail();
        renderPressChart();
        renderBenchmark();
        renderLadderRows();
        setRailReadout(railHoverIndex);
        renderHealth();
    }

    async function fetchProgress() {
        const now = Date.now();
        // /progress may not exist on this server build. Back off instead of
        // hammering it once it has answered badly.
        if (!progressState.available && progressState.lastAttempt
            && now - progressState.lastAttempt < PROGRESS_RETRY_INTERVAL) {
            return null;
        }
        progressState.lastAttempt = now;
        try {
            const response = await fetch(api('/progress'));
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            applyProgressPayload(payload);
            return payload;
        } catch (error) {
            progressState.available = false;
            return null;
        }
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

        const preferLiveArtifacts =
            String(visuals.annotated_frame_path || '').includes('live_frame_annotated') ||
            String(visuals.raw_frame_path || '').includes('live_frame');
        const annotatedArtifactUrl = preferLiveArtifacts
            ? (artifactUrls.live_frame_annotated || artifactUrls.latest_frame_annotated)
            : artifactUrls.latest_frame_annotated;
        const rawArtifactUrl = preferLiveArtifacts
            ? (artifactUrls.live_frame || artifactUrls.latest_frame)
            : artifactUrls.latest_frame;

        if (annotatedArtifactUrl) {
            schedulePreload(
                'annotated',
                els.annotatedFrame,
                withCacheBust(annotatedArtifactUrl, visuals.frame_timestamp),
            );
        }
        if (rawArtifactUrl) {
            schedulePreload(
                'raw',
                els.rawFrame,
                withCacheBust(rawArtifactUrl, visuals.frame_timestamp),
            );
        }

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
        recordObservationSample(payload);
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
        recordHistoryHealth(payload.events || []);
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
                await Promise.all([
                    fetchDashboardState(),
                    fetchDashboardHistory(),
                    fetchSaves(),
                    fetchProgress(),
                ]);
                renderCampaign();
            } catch (error) {
                setStatus(false, 'Server unavailable');
                renderCampaign();
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
        jumpStreamToLive();
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
        jumpStreamToLive();
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

    function setSteerStatus(text, state = 'idle') {
        els.piSteerStatus.textContent = text;
        els.piSteerStatus.dataset.state = state;
    }

    function applySteerAvailability(supervisor) {
        const live = STEERABLE_PI_STATUSES.has(String(supervisor.status || '').toLowerCase());
        els.piSteerInput.disabled = !live;
        els.piSteerButton.disabled = !live;
        // Only rewrite the hint when liveness flips, so a rejection stays readable.
        if (live === steerLive) return;
        steerLive = live;
        setSteerStatus(
            live
                ? 'Delivered at the next tool-call boundary.'
                : 'No live session \u2014 start or continue Pi to send a message.',
        );
    }

    async function sendSteer() {
        const message = els.piSteerInput.value.trim();
        if (!message) {
            setSteerStatus('Type a message first.', 'error');
            return;
        }
        els.piSteerButton.disabled = true;
        setSteerStatus('Sending\u2026');
        try {
            const payload = await postJson('/supervisor/steer', { message });
            // Nothing is inserted into the log here: the entry arrives over the
            // stream, so what you read is what Pi was actually handed.
            els.piSteerInput.value = '';
            jumpStreamToLive();
            const behavior = payload.entry?.streaming_behavior || 'steer';
            setSteerStatus(
                behavior === 'followUp'
                    ? 'Queued \u2014 Pi takes it when the turn ends.'
                    : 'Steered \u2014 Pi takes it at the next tool-call boundary.',
                'sent',
            );
            await refreshAll();
        } catch (error) {
            // Keep the text: a rejected message must not vanish from the box.
            setSteerStatus(String(error.message || error), 'error');
        } finally {
            els.piSteerButton.disabled = !steerLive;
        }
    }

    function onSteerKeydown(event) {
        if (event.key !== 'Enter' || event.shiftKey) return;
        event.preventDefault();
        sendSteer();
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
            healthState.reloads += 1;
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

        if (event.type === 'pi_stream_entry') {
            onStreamEntry(event.entry);
            return;
        }

        if (event.type === 'pi_prompt_sent') {
            els.piControlStatus.textContent = event.resume
                ? 'Sent auto-continue prompt to Pi.'
                : 'Sent launch prompt to Pi.';
            syncStream();
            return;
        }

        // Thinking, text and transcript now arrive as pi_stream_entry updates.
        if (
            event.type === 'pi_transcript' ||
            event.type === 'pi_text_delta' ||
            event.type === 'pi_thinking_delta'
        ) {
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
            els.piControlStatus.textContent = `► ${truncate(event.summary || 'Pi turn launched.', 220)}`;
            syncStream();
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
            syncStream({ full: !streamEntries.length });
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
        initStreamControls();
        buildRail();
        buildLadderRows();
        buildBenchmark();
        buildHealthStrip();
        renderCampaign();
        els.piStartButton.addEventListener('click', startSupervisor);
        els.piContinueButton.addEventListener('click', continueSupervisor);
        els.piStopButton.addEventListener('click', stopSupervisor);
        els.piSteerButton.addEventListener('click', sendSteer);
        els.piSteerInput.addEventListener('keydown', onSteerKeydown);
        applySteerAvailability({});
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
        syncStream({ full: true });
        connectWS();
        pollTimer = window.setInterval(() => {
            refreshAll().catch(() => {});
            syncStream();
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
