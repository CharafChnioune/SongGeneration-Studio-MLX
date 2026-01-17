// SongGeneration Studio - Main App Component
// Dependencies: constants.js, icons.js, api.js, hooks.js, components.js (loaded before this file)

var App = () => {
    // Core state
    const [activeTab, setActiveTab] = useState('create');
    const [sections, setSections] = useState(DEFAULT_SECTIONS);
    const [title, setTitle] = useState('My New Song');
    const [gender, setGender] = useState('female');
    const [genres, setGenres] = useState([]);
    const [moods, setMoods] = useState([]);
    const [timbres, setTimbres] = useState([]);
    const [instruments, setInstruments] = useState([]);
    const [customStyle, setCustomStyle] = useState('');
    const [bpm, setBpm] = useState(120);
    const [outputMode, setOutputMode] = useState(localStorage.getItem('output_mode') || 'mixed');
    const [aiProvider, setAiProvider] = useState(localStorage.getItem('ai_provider') || 'lmstudio');
    const [aiBaseUrl, setAiBaseUrl] = useState(localStorage.getItem('ai_base_url') || AI_PROVIDER_DEFAULTS.lmstudio.baseUrl);
    const [aiModel, setAiModel] = useState(localStorage.getItem('ai_model') || '');
    const [aiPrompt, setAiPrompt] = useState('');
    const [aiLanguage, setAiLanguage] = useState(localStorage.getItem('ai_language') || 'English');
    const [aiLength, setAiLength] = useState('medium');
    const [aiBusy, setAiBusy] = useState(false);
    const [aiError, setAiError] = useState('');

    // Generation state
    const [generating, setGenerating] = useState(false);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState('');
    const [error, setError] = useState(null);
    const [currentGenId, setCurrentGenId] = useState(null);
    const [currentGenPayload, setCurrentGenPayload] = useState(null);
    const [elapsedTime, setElapsedTime] = useState(0);
    const [estimatedTime, setEstimatedTime] = useState(null);

    // Data state
    const [library, setLibrary] = useState([]);
    const [queue, setQueue] = useState([]);
    const [gpuInfo, setGpuInfo] = useState(null);
    const [timingStats, setTimingStats] = useState(null);

    // Reference audio
    const [refId, setRefId] = useState(null);
    const hasReference = Boolean(refId);

    // UI state
    const [showAddMenu, setShowAddMenu] = useState(false);
    const [addMenuPos, setAddMenuPos] = useState({ x: 0, y: 0 });
    const [showModelManager, setShowModelManager] = useState(false);
    const [dragId, setDragId] = useState(null);
    const [dragOverId, setDragOverId] = useState(null);

    // Refs
    const pollRef = useRef(null);
    const timerRef = useRef(null);
    const addBtnRef = useRef(null);
    const queueRef = useRef(queue);  // Track queue for cleanup transitions
    const cleanupRef = useRef(null);  // Ref to hold cleanup function for stable access
    const transitionLockRef = useRef(false);  // Prevent restore effect during transitions

    // Custom hooks
    const modelState = useModels();
    const audioPlayer = useAudioPlayer(library);
    const estimateTime = useTimeEstimation(timingStats);
    const [leftHover, leftHoverHandlers] = useHover();
    const [mainHover, mainHoverHandlers] = useHover();
    const [rightHover, rightHoverHandlers] = useHover();
    const [libraryHover, libraryHoverHandlers] = useHover();

    // API helpers
    const loadLibrary = async () => { try { setLibrary(await fetchLibrary()); } catch (e) { console.error(e); } };
    const loadQueue = async () => { try { setQueue(await fetchQueue()); } catch (e) { console.error(e); } };
    const loadGpuInfo = async () => { try { setGpuInfo(await fetchGpuInfo()); } catch (e) { console.error(e); } };
    const loadTimingStats = async () => { try { setTimingStats(await fetchTimingStats()); } catch (e) { console.error(e); } };

    const formatDownloadDetails = (model) => {
        if (!model) return '';
        const parts = [];
        const totalGb = model.download_total_gb || model.total_gb || model.size_gb;
        if (totalGb) {
            const downloaded = Number(model.downloaded_gb || 0).toFixed(2);
            const total = Number(totalGb).toFixed(1);
            parts.push(`${downloaded} / ${total} GB`);
        }
        if (model.speed_mbps && model.speed_mbps > 0) {
            parts.push(`${Number(model.speed_mbps).toFixed(1)} MB/s`);
        }
        if (model.eta_seconds && model.eta_seconds > 0) {
            parts.push(`ETA ${formatEta(model.eta_seconds)}`);
        }
        if (model.download_stage) {
            parts.unshift(model.download_stage === 'runtime' ? 'Runtime assets' : 'Model files');
        }
        return parts.join(' | ');
    };

    useEffect(() => { localStorage.setItem('output_mode', outputMode); }, [outputMode]);
    useEffect(() => {
        const defaults = AI_PROVIDER_DEFAULTS[aiProvider];
        if (defaults) setAiBaseUrl(defaults.baseUrl);
    }, [aiProvider]);
    useEffect(() => {
        localStorage.setItem('ai_provider', aiProvider);
        localStorage.setItem('ai_base_url', aiBaseUrl);
        localStorage.setItem('ai_model', aiModel);
        localStorage.setItem('ai_language', aiLanguage);
    }, [aiProvider, aiBaseUrl, aiModel, aiLanguage]);


    // Load initial data
    useEffect(() => {
        modelState.loadModels();
        loadLibrary();
        loadGpuInfo();
        loadQueue();
        loadTimingStats();
    }, []);

    // Refresh GPU/models when page becomes visible (user returns to tab)
    useEffect(() => {
        const handleVisibilityChange = () => {
            if (document.visibilityState === 'visible') {
                console.log('[APP] Page visible - refreshing GPU info and models');
                loadGpuInfo();
                modelState.loadModels();  // This also refreshes GPU on backend
            }
        };
        document.addEventListener('visibilitychange', handleVisibilityChange);
        return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
    }, []);

    // Background sync ref
    const bgSyncRef = useRef(null);

    // Keep queueRef in sync with queue state (for use in cleanup without stale closures)
    useEffect(() => { queueRef.current = queue; }, [queue]);

    // Cleanup on unmount
    useEffect(() => () => {
        pollRef.current && clearInterval(pollRef.current);
        timerRef.current && clearInterval(timerRef.current);
        bgSyncRef.current && clearInterval(bgSyncRef.current);
    }, []);

    // Background sync - periodically check for new/completed generations
    // This handles cases where SSE is unavailable or state gets out of sync
    useEffect(() => {
        const bgSync = async () => {
            // Skip if we're actively generating (poll handles that)
            if (generating && currentGenId) return;

            try {
                const lib = await fetchLibrary();
                const runningGen = lib.find(item => ['generating', 'processing'].includes(item.status));

                if (runningGen && !generating && !currentGenId) {
                    // Found a running generation we're not tracking - sync state
                    console.log('[BG-SYNC] Found running generation:', runningGen.id);
                    setLibrary(lib);
                } else if (!runningGen && library.some(l => ['generating', 'processing'].includes(l.status))) {
                    // Generation finished but we didn't catch it - refresh library
                    console.log('[BG-SYNC] Generation completed, refreshing');
                    setLibrary(lib);
                }
            } catch (e) { /* ignore */ }
        };

        // Run background sync every 10 seconds (reduced frequency)
        bgSyncRef.current = setInterval(bgSync, 10000);
        return () => clearInterval(bgSyncRef.current);
    }, [generating, currentGenId, library]);

    // Model selection is now handled by backend (single source of truth)
    // The backend returns 'default' which is the best ready model for current VRAM
    // This is set automatically in hooks.js when loadModels() is called

    // Output mode is user-controlled; do not auto-switch.

    // Restore running generation on page load or when queue item starts
    useEffect(() => {
        console.log('[RESTORE-EFFECT] Triggered - currentGenId:', currentGenId, 'generating:', generating, 'library count:', library.length, 'transitionLock:', transitionLockRef.current);

        // Skip if transition is in progress (cleanupGeneration handles this)
        if (transitionLockRef.current) {
            console.log('[RESTORE-EFFECT] Skipping - transition lock active');
            return;
        }

        // Skip if already tracking a generation
        if (currentGenId) {
            console.log('[RESTORE-EFFECT] Skipping - already tracking:', currentGenId);
            return;
        }

        const runningGen = library.find(item => ['generating', 'processing', 'pending'].includes(item.status));
        console.log('[RESTORE-EFFECT] Looking for running gen in library:', runningGen ? `Found ${runningGen.id} (${runningGen.status})` : 'None found');

        if (!runningGen) {
            console.log('[RESTORE-EFFECT] No running generation in library, exiting');
            return;
        }

        // Note: We no longer skip on `generating && !currentGenId` because that's now
        // our transitional state showing "Starting next song..." - we want to proceed
        // and update with the actual generation data

        console.log('[RESTORE-EFFECT] === SETTING UP TRACKING ===');
        console.log('[RESTORE-EFFECT] Generation ID:', runningGen.id);
        console.log('[RESTORE-EFFECT] Status:', runningGen.status);
        console.log('[RESTORE-EFFECT] Elapsed seconds:', runningGen.elapsed_seconds);
        console.log('[RESTORE-EFFECT] Title:', runningGen.title);

        const meta = runningGen.metadata || {};
        const payload = { title: meta.title || runningGen.title || 'Untitled', model: meta.model || 'songgeneration_base', sections: meta.sections || 5, ...meta };

        console.log('[RESTORE-EFFECT] Setting state: generating=true, currentGenId=' + runningGen.id);
        setGenerating(true);
        setCurrentGenId(runningGen.id);
        setCurrentGenPayload(payload);
        setStatus(runningGen.message || (runningGen.status === 'pending' ? 'Starting...' : 'Generating...'));
        setProgress(runningGen.progress || 0);

        // Only set elapsed time and start timer if generation has actually started (not pending)
        const hasStarted = runningGen.status !== 'pending' && typeof runningGen.elapsed_seconds === 'number';
        console.log('[RESTORE-EFFECT] hasStarted:', hasStarted, '- Setting elapsedTime:', hasStarted ? runningGen.elapsed_seconds : 0);
        setElapsedTime(hasStarted ? runningGen.elapsed_seconds : 0);
        setEstimatedTime(estimateTime(payload.model, Array.isArray(payload.sections) ? payload.sections : [], Boolean(payload.reference_audio_id)));

        // Clear any existing intervals before setting new ones
        console.log('[RESTORE-EFFECT] Clearing existing intervals - timerRef:', !!timerRef.current, 'pollRef:', !!pollRef.current);
        if (timerRef.current) clearInterval(timerRef.current);
        if (pollRef.current) clearInterval(pollRef.current);

        const genId = runningGen.id;
        // Only start elapsed time counter if generation has actually started
        if (hasStarted) {
            console.log('[RESTORE-EFFECT] Starting elapsed time counter');
            timerRef.current = setInterval(() => setElapsedTime(prev => prev + 1), 1000);
        }

        // Poll function will be called via closure - capture genId explicitly
        console.log('[RESTORE-EFFECT] Starting 2-second status poll for:', genId);
        pollRef.current = setInterval(async () => {
            try {
                console.log('[RESTORE-POLL] Fetching status for:', genId);
                const d = await fetchGeneration(genId);
                if (!d) {
                    // Generation not found (404) - clean up stale state
                    console.log('[RESTORE-POLL] Generation not found (404), cleaning up:', genId);
                    clearInterval(pollRef.current);
                    clearInterval(timerRef.current);
                    pollRef.current = null;
                    timerRef.current = null;
                    setCurrentGenId(null);
                    setCurrentGenPayload(null);
                    setEstimatedTime(null);
                    setElapsedTime(0);
                    setGenerating(false);
                    await Promise.all([loadLibrary(), loadQueue()]);
                    return;
                }
                console.log('[RESTORE-POLL] Status:', d.status, 'Progress:', d.progress, 'Elapsed:', d.elapsed_seconds);
                setProgress(d.progress);
                setStatus(d.message);
                // Only update elapsed time and start timer if generation has actually started
                if (d.status !== 'pending' && typeof d.elapsed_seconds === 'number') {
                    setElapsedTime(d.elapsed_seconds);
                    // Start timer if not already running
                    if (!timerRef.current) {
                        console.log('[RESTORE-POLL] Starting timer (was not running)');
                        timerRef.current = setInterval(() => setElapsedTime(prev => prev + 1), 1000);
                    }
                }
                if (['completed', 'failed', 'stopped'].includes(d.status)) {
                    console.log('[RESTORE-POLL] === GENERATION FINISHED ===');
                    console.log('[RESTORE-POLL] Final status:', d.status);
                    if (d.status === 'failed') setError(d.message);
                    // Call cleanup via ref to ensure queue transitions work properly
                    console.log('[RESTORE-POLL] Calling cleanupGeneration via cleanupRef...');
                    if (cleanupRef.current) {
                        cleanupRef.current();
                    } else {
                        console.error('[RESTORE-POLL] cleanupRef.current is null!');
                        // Fallback: minimal cleanup
                        clearInterval(pollRef.current);
                        clearInterval(timerRef.current);
                        pollRef.current = null;
                        timerRef.current = null;
                        setCurrentGenId(null);
                        setCurrentGenPayload(null);
                        setGenerating(false);
                    }
                }
            } catch (e) { console.error('[RESTORE-POLL] Error:', e); }
        }, 2000);
        console.log('[RESTORE-EFFECT] === TRACKING SETUP COMPLETE ===');
    }, [library, currentGenId, estimateTime]);

    // Helper to set up tracking for a generation (used by cleanup and restore effect)
    const setupGenerationTracking = useCallback(async (gen, freshLibrary) => {
        console.log('[SETUP-TRACKING] === SETTING UP TRACKING FOR:', gen.id, '===');

        const meta = gen.metadata || {};
        // Get model from multiple sources (queue items have model directly, library items have it in metadata)
        const model = meta.model || gen.model || 'songgeneration_base';
        // Get sections array - could be in metadata or directly on gen (for queue items)
        const sections = Array.isArray(meta.sections) ? meta.sections : (Array.isArray(gen.sections) ? gen.sections : []);

        const payload = {
            title: meta.title || gen.title || 'Untitled',
            model,
            sections,
            ...meta
        };

        // Fetch fresh queue AND timing stats to ensure we have up-to-date estimate
        const [freshQueue, freshTimingStats] = await Promise.all([
            fetchQueue(),
            fetchTimingStats()
        ]);

        // Update timing stats state so future estimates use fresh data
        setTimingStats(freshTimingStats);

        // Set all state in one batch - library AND queue together
        setLibrary(freshLibrary);
        setQueue(freshQueue);
        setCurrentGenId(gen.id);
        setCurrentGenPayload(payload);
        setStatus(gen.message || (gen.status === 'pending' ? 'Starting...' : 'Generating...'));
        setProgress(gen.progress || 0);

        const hasStarted = gen.status !== 'pending' && typeof gen.elapsed_seconds === 'number';
        setElapsedTime(hasStarted ? gen.elapsed_seconds : 0);
        // Use fresh timing stats for the estimate - use metadata's stored values as fallback
        const numSections = sections.length || meta.num_sections || 5;
        const totalLyrics = sections.length > 0
            ? sections.reduce((acc, s) => acc + (s.lyrics || '').length, 0)
            : (meta.total_lyrics_length || 0);
        setEstimatedTime(calculateEstimateFromValues(freshTimingStats, model, numSections, totalLyrics, Boolean(meta.reference_audio_id || gen.reference_audio_id)));

        // Start timer if generation has started
        if (hasStarted && !timerRef.current) {
            console.log('[SETUP-TRACKING] Starting elapsed time counter');
            timerRef.current = setInterval(() => setElapsedTime(prev => prev + 1), 1000);
        }

        // Start polling for this generation
        const genId = gen.id;
        console.log('[SETUP-TRACKING] Starting 2-second poll for:', genId);
        pollRef.current = setInterval(async () => {
            try {
                console.log('[CLEANUP-POLL] Fetching status for:', genId);
                const d = await fetchGeneration(genId);
                if (!d) {
                    console.log('[CLEANUP-POLL] Generation not found (404), cleaning up');
                    clearInterval(pollRef.current);
                    clearInterval(timerRef.current);
                    pollRef.current = null;
                    timerRef.current = null;
                    setCurrentGenId(null);
                    setCurrentGenPayload(null);
                    setEstimatedTime(null);
                    setElapsedTime(0);
                    setGenerating(false);
                    transitionLockRef.current = false;
                    await Promise.all([loadLibrary(), loadQueue()]);
                    return;
                }
                console.log('[CLEANUP-POLL] Status:', d.status, 'Progress:', d.progress, 'Elapsed:', d.elapsed_seconds);
                setProgress(d.progress);
                setStatus(d.message);
                if (d.status !== 'pending' && typeof d.elapsed_seconds === 'number') {
                    setElapsedTime(d.elapsed_seconds);
                    if (!timerRef.current) {
                        console.log('[CLEANUP-POLL] Starting timer (was not running)');
                        timerRef.current = setInterval(() => setElapsedTime(prev => prev + 1), 1000);
                    }
                }
                if (['completed', 'failed', 'stopped'].includes(d.status)) {
                    console.log('[CLEANUP-POLL] === GENERATION FINISHED ===');
                    if (d.status === 'failed') setError(d.message);
                    if (cleanupRef.current) {
                        cleanupRef.current();
                    }
                }
            } catch (e) {
                console.error('[CLEANUP-POLL] Error:', e);
            }
        }, 2000);

        console.log('[SETUP-TRACKING] === TRACKING SETUP COMPLETE ===');
    }, []);

    const cleanupGeneration = useCallback(async () => {
        console.log('[CLEANUP] === CLEANUP GENERATION CALLED ===');

        // Set transition lock IMMEDIATELY to prevent restore effect interference
        transitionLockRef.current = true;
        console.log('[CLEANUP] Transition lock SET');

        console.log('[CLEANUP] Clearing intervals - pollRef:', !!pollRef.current, 'timerRef:', !!timerRef.current);
        clearInterval(pollRef.current);
        clearInterval(timerRef.current);
        pollRef.current = null;
        timerRef.current = null;

        // Check if there are queued items BEFORE we load fresh data
        const currentQueue = queueRef.current;
        const hadQueuedItems = currentQueue && currentQueue.length > 0;
        console.log('[CLEANUP] Queue state - items:', currentQueue?.length || 0, 'hadQueuedItems:', hadQueuedItems);

        // Clear generation state - but keep generating=true if we have queue items
        console.log('[CLEANUP] Clearing generation state');
        setCurrentGenId(null);
        setCurrentGenPayload(null);
        setEstimatedTime(null);
        setElapsedTime(0);
        setProgress(0);
        setStatus('');

        if (!hadQueuedItems) {
            console.log('[CLEANUP] No queued items - setting generating=false');
            setGenerating(false);
            transitionLockRef.current = false;
            console.log('[CLEANUP] Transition lock CLEARED (no queue)');

            // Load fresh data
            await Promise.all([loadLibrary(), loadQueue(), loadTimingStats(), loadGpuInfo()]);
            modelState.loadModels();
            console.log('[CLEANUP] === CLEANUP COMPLETE (no queue) ===');
            return;
        }

        console.log('[CLEANUP] Queue has items - looking for next generation...');

        // Fetch fresh library to find next generation
        const freshLib = await fetchLibrary();
        const nextGen = freshLib.find(item => ['generating', 'processing', 'pending'].includes(item.status));

        if (nextGen) {
            // Found next generation - set up tracking DIRECTLY (not via restore effect)
            console.log('[CLEANUP] Found next gen:', nextGen.id, nextGen.status, '- setting up tracking directly');
            try {
                await setupGenerationTracking(nextGen, freshLib);
            } catch (e) {
                console.error('[CLEANUP] Error in setupGenerationTracking:', e);
                // On error, fall back to clean state
                setGenerating(false);
                await Promise.all([loadLibrary(), loadQueue()]);
            }

            // Load other data in background
            loadTimingStats();
            loadGpuInfo();
            modelState.loadModels();

            // Clear lock AFTER all state is set (always clear, even on error)
            transitionLockRef.current = false;
            console.log('[CLEANUP] Transition lock CLEARED (next gen found)');
        } else {
            // Backend hasn't started next gen yet - poll until it does
            console.log('[CLEANUP] Next gen not in library yet, starting transition poll');

            // Update BOTH library AND queue state now to prevent ghost items
            // (queue item may have been popped by backend but our state is stale)
            const freshQueue = await fetchQueue();
            setLibrary(freshLib);
            setQueue(freshQueue);

            pollRef.current = setInterval(async () => {
                try {
                    console.log('[TRANSITION-POLL] Checking for next generation...');
                    const lib = await fetchLibrary();
                    const running = lib.find(item => ['generating', 'processing', 'pending'].includes(item.status));

                    if (running) {
                        console.log('[TRANSITION-POLL] Found next gen:', running.id, running.status);
                        clearInterval(pollRef.current);
                        pollRef.current = null;

                        // Set up tracking DIRECTLY - await to ensure state is set before clearing lock
                        try {
                            await setupGenerationTracking(running, lib);
                        } catch (e) {
                            console.error('[TRANSITION-POLL] Error in setupGenerationTracking:', e);
                            // On error, fall back to clean state
                            setGenerating(false);
                            setLibrary(lib);
                            await loadQueue();
                        }

                        // Clear lock after state is set (always clear, even on error)
                        transitionLockRef.current = false;
                        console.log('[TRANSITION-POLL] Transition lock CLEARED');
                    } else {
                        // Check if queue is now empty
                        const q = await fetchQueue();
                        console.log('[TRANSITION-POLL] No running gen, queue length:', q.length);
                        if (q.length === 0) {
                            console.log('[TRANSITION-POLL] Queue empty - clearing generating state');
                            clearInterval(pollRef.current);
                            pollRef.current = null;
                            setGenerating(false);
                            setLibrary(lib);
                            setQueue(q);
                            transitionLockRef.current = false;
                            console.log('[TRANSITION-POLL] Transition lock CLEARED (queue empty)');
                        }
                    }
                } catch (e) {
                    console.error('[TRANSITION-POLL] Error:', e);
                }
            }, 1000);
        }

        console.log('[CLEANUP] === CLEANUP COMPLETE ===');
    }, [setupGenerationTracking]);

    const poll = useCallback(async (id) => {
        try {
            console.log('[POLL] Fetching status for:', id);
            const d = await fetchGeneration(id);
            if (!d) {
                console.log('[POLL] Generation not found (404), calling cleanup');
                cleanupGeneration();
                return;
            }
            console.log('[POLL] Status:', d.status, 'Progress:', d.progress, 'Elapsed:', d.elapsed_seconds, 'Message:', d.message);
            setProgress(d.progress);
            setStatus(d.message);
            if (typeof d.elapsed_seconds === 'number') setElapsedTime(d.elapsed_seconds);
            if (['completed', 'failed', 'stopped'].includes(d.status)) {
                console.log('[POLL] === GENERATION FINISHED ===');
                console.log('[POLL] Final status:', d.status);
                if (d.status === 'failed') setError(d.message);
                console.log('[POLL] Calling cleanupGeneration...');
                cleanupGeneration();
            }
        } catch (e) { console.error('[POLL] Error:', e); }
    }, [cleanupGeneration]);

    // Keep cleanupRef in sync with cleanupGeneration (for use in restore effect without stale closures)
    useEffect(() => { cleanupRef.current = cleanupGeneration; }, [cleanupGeneration]);

    const createPayload = () => ({
        title,
        sections: sections.map(s => ({ type: s.type, lyrics: s.lyrics || null })),
        gender,
        genre: genres.join(', '),
        emotion: moods.join(', '),
        timbre: timbres.join(', '),
        instruments: instruments.join(', '),
        custom_style: customStyle,
        bpm,
        output_mode: outputMode,
        model: modelState.selectedModel,
        reference_audio_id: refId || null,
    });

    const splitTagList = (value) => {
        if (!value) return [];
        if (Array.isArray(value)) return value.map(v => String(v).trim()).filter(Boolean);
        return String(value).split(',').map(v => v.trim()).filter(Boolean);
    };

    const normalizeAiSections = (sectionsPayload) => {
        if (!Array.isArray(sectionsPayload) || sectionsPayload.length === 0) return null;
        return sectionsPayload.map((section, idx) => ({
            id: `${Date.now()}-${idx}`,
            type: section.type || 'verse',
            lyrics: section.lyrics || '',
        }));
    };

    const runAIAssist = async () => {
        setAiError('');
        if (!aiModel || !aiModel.trim()) {
            setAiError('Add a model name for the AI provider.');
            return;
        }
        if (!aiPrompt || !aiPrompt.trim()) {
            setAiError('Add a prompt or theme for the AI.');
            return;
        }
        setAiBusy(true);
        try {
            const payload = {
                provider: aiProvider,
                base_url: aiBaseUrl,
                model: aiModel,
                prompt: aiPrompt,
                language: aiLanguage,
                length: aiLength,
            };
            const result = await requestAIAssist(payload);
            if (result.gender) setGender(result.gender);
            if (result.genre) setGenres([result.genre]);
            if (result.emotion) setMoods(splitTagList(result.emotion));
            if (result.timbre) setTimbres(splitTagList(result.timbre));
            if (result.instruments) setInstruments(splitTagList(result.instruments).slice(-1));
            if (result.bpm) setBpm(Number(result.bpm));
            if (result.custom_style) setCustomStyle(result.custom_style);

            const updatedSections = normalizeAiSections(result.sections);
            if (updatedSections) setSections(updatedSections);
        } catch (e) {
            setAiError(e.message || 'AI assist failed.');
        } finally {
            setAiBusy(false);
        }
    };

    const doStartGeneration = async (payload) => {
        console.log('[START-GEN] === STARTING NEW GENERATION ===');
        console.log('[START-GEN] Title:', payload.title);
        console.log('[START-GEN] Model:', payload.model);
        console.log('[START-GEN] Sections:', payload.sections?.length);
        setProgress(0); setStatus('Starting...'); setError(null);
        setCurrentGenPayload(payload);
        // Fetch fresh timing stats for accurate estimate
        const freshStats = await fetchTimingStats();
        setTimingStats(freshStats);
        const baseEstimate = calculateEstimate(freshStats, payload.model, payload.sections, Boolean(payload.reference_audio_id));
        setEstimatedTime(baseEstimate);
        setElapsedTime(0);
        console.log('[START-GEN] Starting elapsed time counter');
        timerRef.current = setInterval(() => setElapsedTime(prev => prev + 1), 1000);
        try {
            console.log('[START-GEN] Calling API to start generation...');
            const { generation_id } = await startGeneration(payload);
            console.log('[START-GEN] API returned generation_id:', generation_id);
            setCurrentGenId(generation_id);
            console.log('[START-GEN] Starting 2-second poll for:', generation_id);
            pollRef.current = setInterval(() => poll(generation_id), 2000);
            console.log('[START-GEN] === GENERATION STARTED ===');
        } catch (e) {
            console.error('[START-GEN] Error starting generation:', e.message);
            setGenerating(false); setCurrentGenId(null); setCurrentGenPayload(null);
            clearInterval(timerRef.current);
            if (e.message.includes('409') || e.message.includes('already in progress')) {
                console.log('[START-GEN] Generation already in progress, loading library');
                loadLibrary();
            }
            else setError(e.message);
        }
    };

    const generate = async () => {
        console.log('[GENERATE] Generate button clicked');
        console.log('[GENERATE] hasReadyModel:', modelState.hasReadyModel, 'models count:', modelState.models.length);
        console.log('[GENERATE] Currently generating:', generating);
        if (!modelState.hasReadyModel || modelState.models.length === 0) {
            console.log('[GENERATE] No ready models, showing error');
            setError("No models downloaded. Please download a model first."); return;
        }
        let modelToUse = modelState.selectedModel;
        if (!modelState.models.some(m => m.id === modelToUse && m.status === 'ready')) {
            const firstReady = modelState.models.find(m => m.status === 'ready');
            if (firstReady) { modelToUse = firstReady.id; modelState.setSelectedModel(modelToUse); }
            else { setError("No models ready."); return; }
        }
        const payload = { ...createPayload(), model: modelToUse };
        console.log('[GENERATE] Payload created with title:', payload.title, 'model:', modelToUse);
        if (generating) {
            console.log('[GENERATE] Already generating, adding to queue...');
            await addToQueue(payload);
            await loadQueue();
            console.log('[GENERATE] Added to queue');
        }
        else {
            console.log('[GENERATE] Not generating, starting new generation');
            setGenerating(true);
            doStartGeneration(payload);
        }
    };

    // Section handlers
    const addSection = (base) => {
        const cfg = SECTION_TYPES[base];
        const type = cfg?.hasDuration ? `${base}-short` : base;
        setSections([...sections, { id: Date.now().toString(), type, lyrics: '' }]);
        setShowAddMenu(false);
    };
    const removeSection = (id) => setSections(sections.filter(s => s.id !== id));
    const updateSection = (id, updates) => setSections(sections.map(s => s.id === id ? { ...s, ...updates } : s));

    // Drag handlers
    const handleDragStart = (e, id) => { e.dataTransfer.setData('text/plain', id); setDragId(id); };
    const handleDragEnd = () => { setDragId(null); setDragOverId(null); setDropIndex(null); };

    // Track drop insertion position
    const [dropIndex, setDropIndex] = useState(null);

    // Calculate drop position based on mouse position
    const handleDragOverWithPosition = (e, targetId, targetIndex) => {
        e.preventDefault();
        if (dragId === targetId) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const insertBefore = e.clientX < rect.left + rect.width / 2;
        const newDropIndex = insertBefore ? targetIndex : targetIndex + 1;
        if (newDropIndex !== dropIndex) {
            setDropIndex(newDropIndex);
            setDragOverId(targetId);
        }
    };

    const handleDropWithPosition = (e) => {
        e.preventDefault();
        if (dragId === null || dropIndex === null) { handleDragEnd(); return; }
        const dragIndex = sections.findIndex(s => s.id === dragId);
        if (dragIndex === -1) { handleDragEnd(); return; }
        let adjustedDropIndex = dropIndex;
        if (dragIndex < dropIndex) adjustedDropIndex = dropIndex - 1;
        if (dragIndex !== adjustedDropIndex) {
            const newSections = [...sections];
            const [removed] = newSections.splice(dragIndex, 1);
            newSections.splice(adjustedDropIndex, 0, removed);
            setSections(newSections);
        }
        handleDragEnd();
    };

    const toggleAddMenu = (e) => {
        e.stopPropagation();
        if (!showAddMenu) {
            const rect = e.currentTarget.getBoundingClientRect();
            const openUpward = window.innerHeight - rect.bottom < 280;
            setAddMenuPos({ x: rect.left, y: openUpward ? rect.top : rect.bottom + 8, openUpward });
        }
        setShowAddMenu(!showAddMenu);
    };

    // Close popup on outside click
    useEffect(() => {
        const handleClick = (e) => { if (showAddMenu && addBtnRef.current && !addBtnRef.current.contains(e.target)) setShowAddMenu(false); };
        document.addEventListener('click', handleClick);
        return () => document.removeEventListener('click', handleClick);
    }, [showAddMenu]);

    // Duration widths for resizable sections
    const DURATION_WIDTHS = { short: 68, medium: 92 };
    const DURATION_THRESHOLDS = { medium: 82 };

    // Resize state
    const [resizingId, setResizingId] = useState(null);
    const [resizeStartX, setResizeStartX] = useState(0);
    const [resizeStartWidth, setResizeStartWidth] = useState(0);

    const handleResizeStart = (e, sectionId, currentWidth) => {
        e.preventDefault();
        e.stopPropagation();
        setResizingId(sectionId);
        setResizeStartX(e.clientX);
        setResizeStartWidth(currentWidth);
    };

    const updateDurationFromWidth = (sectionId, newWidth) => {
        const newDuration = newWidth >= DURATION_THRESHOLDS.medium ? 'medium' : 'short';
        setSections(prev => prev.map(s => {
            if (s.id !== sectionId) return s;
            const { base, duration } = fromApiType(s.type);
            if (duration === newDuration) return s;
            return { ...s, type: `${base}-${newDuration}` };
        }));
    };

    // Global mouse handlers for resize
    useEffect(() => {
        if (!resizingId) return;
        document.body.style.cursor = 'ew-resize';
        document.body.style.userSelect = 'none';
        const handleMouseMove = (e) => {
            const delta = e.clientX - resizeStartX;
            const newWidth = Math.max(50, Math.min(130, resizeStartWidth + delta));
            updateDurationFromWidth(resizingId, newWidth);
        };
        const handleMouseUp = () => {
            setResizingId(null);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        };
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        };
    }, [resizingId, resizeStartX, resizeStartWidth]);

    return (
        <div style={{ height: '100vh', backgroundColor: '#1e1e1e', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {/* Header */}
            <header style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '20px 24px', maxWidth: '1400px', margin: '0 auto', width: '100%', boxSizing: 'border-box', flexShrink: 0, position: 'relative' }}>
                <div style={{ position: 'absolute', left: '24px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <img src="/static/Logo_1.png" alt="SongGeneration" style={{ height: '36px' }} />
                    <div>
                        <div style={{ fontSize: '18px', fontWeight: '600', color: '#e0e0e0' }}>SongGeneration Studio</div>
                        <div style={{ fontSize: '11px', color: '#666' }}>by Tencent AI Lab</div>
                    </div>
                </div>
                <div style={{ display: 'flex', gap: '4px', backgroundColor: '#282828', padding: '4px', borderRadius: '12px' }}>
                    {['create', 'library'].map(tab => (
                        <button key={tab} onClick={() => { setActiveTab(tab); if (tab === 'library') loadLibrary(); }}
                            style={{ padding: '10px 24px', borderRadius: '10px', border: 'none', fontSize: '14px', fontWeight: '500', cursor: 'pointer', backgroundColor: activeTab === tab ? '#10B981' : 'transparent', color: activeTab === tab ? '#fff' : '#888', transition: 'all 0.15s' }}>
                            {tab.charAt(0).toUpperCase() + tab.slice(1)}
                        </button>
                    ))}
                </div>
                <a href="https://github.com/CharafChnioune/SongGeneration-Studio-MLX" target="_blank" rel="noopener noreferrer"
                    style={{ position: 'absolute', right: '24px', display: 'flex', alignItems: 'center', gap: '6px', color: '#666', textDecoration: 'none', padding: '8px 12px', borderRadius: '8px', transition: 'all 0.15s' }}
                    onMouseEnter={e => { e.currentTarget.style.color = '#fff'; e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.1)'; }}
                    onMouseLeave={e => { e.currentTarget.style.color = '#666'; e.currentTarget.style.backgroundColor = 'transparent'; }}>
                    <GitHubIcon size={18} />
                </a>
            </header>

            {/* Main Content */}
            <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                {/* Create View */}
                <div style={{ display: activeTab === 'create' ? 'flex' : 'none', flex: 1, overflow: 'hidden' }}>
                    <div style={{ display: 'flex', gap: '24px', maxWidth: '1400px', margin: '0 auto', padding: '0 24px 24px 24px', flex: 1, overflow: 'hidden' }}>
                        {/* Left Sidebar */}
                        <aside {...leftHoverHandlers} style={{ width: '320px', flexShrink: 0, display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto', paddingBottom: '100px', paddingRight: '8px', ...getScrollStyle(leftHover) }}>
                            {/* Model Card */}
                            <Card>
                                <CardTitle>Model</CardTitle>
                                {modelState.hasReadyModel ? (
                                    <div style={{ position: 'relative', marginBottom: '12px' }}>
                                        <select className="custom-select input-base" value={modelState.selectedModel} onChange={e => modelState.setSelectedModel(e.target.value)} style={{ paddingRight: '40px', cursor: 'pointer' }}>
                                            {modelState.models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                                        </select>
                                        <div style={{ position: 'absolute', right: '14px', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }}><ChevronIcon /></div>
                                    </div>
                                ) : (
                                    <div style={{
                                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                                        border: '1px solid rgba(245, 158, 11, 0.3)',
                                        borderRadius: '10px',
                                        padding: '12px 14px',
                                        marginBottom: '12px',
                                        color: '#F59E0B',
                                        fontSize: '13px',
                                        fontWeight: '500',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        gap: '8px'
                                    }}>
                                        {modelState.isInitializing ? <><SpinnerIcon size={14} /> Loading...</> : modelState.allModels.some(m => m.status === 'downloading') ? 'Downloading model...' : 'No Models Downloaded'}
                                    </div>
                                )}
                                {modelState.allModels.filter(m => m.status === 'downloading').map(m => {
                                    const details = formatDownloadDetails(m);
                                    const stageLabel = m.download_stage === 'runtime' ? 'Runtime assets' : 'Model files';
                                    const isIndeterminate = !m.progress || m.progress <= 0;
                                    return (
                                        <div key={m.id} style={{ backgroundColor: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.3)', borderRadius: '8px', padding: '10px 12px', marginBottom: '12px' }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                                                <span style={{ fontSize: '12px', color: '#F59E0B', fontWeight: '500' }}>Downloading {stageLabel}</span>
                                                <span style={{ fontSize: '11px', color: '#888' }}>{m.progress || 0}%</span>
                                            </div>
                                            <div className={isIndeterminate ? 'progress-bar-indeterminate' : ''} style={{ height: '6px', backgroundColor: 'rgba(245, 158, 11, 0.2)', borderRadius: '3px', overflow: 'hidden' }}>
                                                {!isIndeterminate && <div style={{ width: `${m.progress || 0}%`, height: '100%', backgroundColor: '#F59E0B', borderRadius: '3px', transition: 'width 0.3s' }} />}
                                            </div>
                                            {details && <div style={{ marginTop: '6px', fontSize: '10px', color: '#C0841A' }}>{details}</div>}
                                            <button onClick={() => modelState.cancelDownload(m.id)} style={{ background: 'none', border: 'none', color: '#EF4444', cursor: 'pointer', padding: '2px 6px', fontSize: '10px', marginTop: '6px' }}>Cancel</button>
                                        </div>
                                    );
                                })}
                                <button onClick={() => setShowModelManager(true)} style={{ width: '100%', padding: '10px', backgroundColor: '#2a2a2a', border: '1px solid #444', borderRadius: '8px', color: '#fff', fontSize: '12px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                                    <SettingsIcon /> Manage Models
                                </button>
                            </Card>

                            {/* AI Assist */}
                            <Card>
                                <CardTitle>AI Assist</CardTitle>
                                <div style={{ marginBottom: '12px' }}>
                                    <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Provider</div>
                                    <select className="custom-select input-base" value={aiProvider} onChange={e => setAiProvider(e.target.value)} style={{ paddingRight: '40px', cursor: 'pointer' }}>
                                        {Object.entries(AI_PROVIDER_DEFAULTS).map(([key, cfg]) => (
                                            <option key={key} value={key}>{cfg.name}</option>
                                        ))}
                                    </select>
                                    <div style={{ fontSize: '10px', color: '#555', marginTop: '6px' }}>{AI_PROVIDER_DEFAULTS[aiProvider]?.help}</div>
                                </div>
                                <div style={{ marginBottom: '12px' }}>
                                    <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Base URL</div>
                                    <input type="text" className="input-base" value={aiBaseUrl} onChange={e => setAiBaseUrl(e.target.value)} placeholder={AI_PROVIDER_DEFAULTS[aiProvider]?.baseUrl} />
                                </div>
                                <div style={{ marginBottom: '12px' }}>
                                    <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Model</div>
                                    <input type="text" className="input-base" value={aiModel} onChange={e => setAiModel(e.target.value)} placeholder="e.g. amsi-fin-mlx" />
                                </div>
                                <div style={{ marginBottom: '12px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                                    <div>
                                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Language</div>
                                        <select className="custom-select input-base" value={aiLanguage} onChange={e => setAiLanguage(e.target.value)} style={{ paddingRight: '40px', cursor: 'pointer' }}>
                                            {['English', 'Chinese'].map(lang => <option key={lang} value={lang}>{lang}</option>)}
                                        </select>
                                    </div>
                                    <div>
                                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Length</div>
                                        <select className="custom-select input-base" value={aiLength} onChange={e => setAiLength(e.target.value)} style={{ paddingRight: '40px', cursor: 'pointer' }}>
                                            {['short', 'medium', 'full'].map(size => <option key={size} value={size}>{size}</option>)}
                                        </select>
                                    </div>
                                </div>
                                <div style={{ marginBottom: '12px' }}>
                                    <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Prompt / Theme</div>
                                    <textarea className="input-base" value={aiPrompt} onChange={e => setAiPrompt(e.target.value)} placeholder="Describe the song idea..." rows={4} style={{ resize: 'vertical' }} />
                                </div>
                                {aiError && <div style={{ color: '#EF4444', fontSize: '11px', marginBottom: '8px' }}>{aiError}</div>}
                                <button onClick={runAIAssist} disabled={aiBusy} style={{ width: '100%', padding: '12px', borderRadius: '10px', border: '1px solid #10B981', backgroundColor: aiBusy ? '#1f3d33' : '#10B98115', color: '#10B981', fontSize: '12px', cursor: aiBusy ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                                    {aiBusy ? <><SpinnerIcon size={14} /> Generating...</> : 'Generate Genre + Mood + Structure + Lyrics'}
                                </button>
                                <div style={{ fontSize: '10px', color: '#555', marginTop: '8px' }}>
                                    The assistant fills tags, builds a structure, and writes lyrics using the paper-aligned format.
                                </div>
                            </Card>

                        {/* Song Settings */}
                        <Card>
                                <CardTitle>Song Settings</CardTitle>
                                <div>
                                    <div style={{ marginBottom: '16px' }}>
                                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Voice</div>
                                        <div style={{ display: 'flex', gap: '8px' }}>
                                            {['female', 'male'].map(g => <button key={g} onClick={() => setGender(g)} style={btnStyle(gender === g, '#3B82F6')}>{g.charAt(0).toUpperCase() + g.slice(1)}</button>)}
                                        </div>
                                    </div>
                                    <div style={{ marginBottom: '16px' }}>
                                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Genre</div>
                                        <MultiSelectWithScroll suggestions={GENRE_SUGGESTIONS} selected={genres} onChange={(next) => setGenres(next.slice(-1))} placeholder="Select genre..." />
                                        <div style={{ fontSize: '10px', color: '#555', marginTop: '6px' }}>Pick one recommended tag for best alignment.</div>
                                    </div>
                                    <div style={{ marginBottom: '16px' }}><div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Mood</div><MultiSelectWithScroll suggestions={MOOD_SUGGESTIONS} selected={moods} onChange={setMoods} placeholder="Select mood..." /></div>
                                    <div style={{ marginBottom: '16px' }}><div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Timbre</div><MultiSelectWithScroll suggestions={TIMBRE_SUGGESTIONS} selected={timbres} onChange={setTimbres} placeholder="Select timbre..." /></div>
                                    <div style={{ marginBottom: '16px' }}>
                                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Instruments</div>
                                        <MultiSelectWithScroll suggestions={INSTRUMENT_SUGGESTIONS} selected={instruments} onChange={(next) => setInstruments(next.slice(-1))} placeholder="Select instruments..." />
                                        <div style={{ fontSize: '10px', color: '#555', marginTop: '6px' }}>Choose one instrument preset; add extras in Custom Style.</div>
                                    </div>
                                    <div style={{ marginBottom: '16px' }}>
                                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Custom Style</div>
                                        <input type="text" value={customStyle} onChange={e => setCustomStyle(e.target.value)} placeholder="e.g. dubstep wobble..." className="input-base" />
                                    </div>
                                    <div>
                                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}><span>BPM</span><span style={{ color: '#10B981', fontWeight: '500' }}>{bpm}</span></div>
                                        <input type="range" min="60" max="180" value={bpm} onChange={e => setBpm(+e.target.value)} />
                                    </div>
                                </div>
                            </Card>

                            {/* Reference Audio */}
                            <Card>
                                <CardTitle>Reference Audio (Optional)</CardTitle>
                                <AudioTrimmer
                                    onAccept={(d) => { setRefId(d.id); }}
                                    onClear={() => { setRefId(null); }}
                                />
                                <div style={{ fontSize: '10px', color: '#555' }}>
                                    The model accepts lyrics + optional text descriptions + optional audio prompts. Only the first 10 seconds are used.
                                </div>
                                {hasReference && (
                                    <div style={{ fontSize: '10px', color: '#C0841A', marginTop: '6px' }}>
                                        Reference audio can be combined with tags, but conflicting prompts may reduce alignment.
                                    </div>
                                )}
                            </Card>

                            {/* Output */}
                            <Card>
                                <CardTitle>Output</CardTitle>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                                    {['mixed', 'vocal', 'bgm', 'separate'].map(mode => (
                                        <button key={mode} onClick={() => setOutputMode(mode)} style={{ padding: '12px', borderRadius: '10px', border: outputMode === mode ? '1px solid #10B981' : '1px solid #3a3a3a', backgroundColor: outputMode === mode ? '#10B98115' : '#1e1e1e', color: outputMode === mode ? '#10B981' : '#777', cursor: 'pointer', fontSize: '13px', fontWeight: '500' }}>
                                            {mode === 'mixed' ? 'Full Song' : mode === 'vocal' ? 'Vocals' : mode === 'bgm' ? 'Instrumental' : 'Separate'}
                                        </button>
                                    ))}
                                </div>
                            </Card>

                            {/* Song Title */}
                            <Card><CardTitle>Song Title</CardTitle><input type="text" className="input-base" value={title} onChange={e => setTitle(e.target.value)} placeholder="Enter song title..." style={{ fontSize: '14px' }} /></Card>

                            {/* Generate Button */}
                            <button onClick={generate} disabled={!modelState.hasReadyModel} style={{ width: '100%', backgroundColor: !modelState.hasReadyModel ? '#444' : generating ? '#6366F1' : '#10B981', color: '#fff', border: 'none', borderRadius: '12px', padding: '16px 24px', fontSize: '15px', fontWeight: '600', cursor: !modelState.hasReadyModel ? 'not-allowed' : 'pointer', opacity: !modelState.hasReadyModel ? 0.6 : 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                                {!modelState.hasReadyModel ? 'Download Model to Generate' : generating ? <><PlusIcon /> Add to Queue {queue.length > 0 && `(${queue.length})`}</> : <><PlayIcon /> Generate</>}
                            </button>
                        </aside>

                        {/* Main Content */}
                        <main {...mainHoverHandlers} style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto', paddingBottom: '100px', paddingRight: '8px', ...getScrollStyle(mainHover) }}>
                            {/* Structure */}
                            <Card>
                                <CardTitle>Structure</CardTitle>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0', overflowX: 'auto', paddingBottom: '4px', minHeight: '28px' }} onDragOver={(e) => e.preventDefault()} onDrop={handleDropWithPosition}>
                                    {sections.map((s, index) => {
                                        const { base, duration } = fromApiType(s.type);
                                        const cfg = SECTION_TYPES[base] || { name: base, color: '#888' };
                                        const isResizable = cfg.hasDuration;
                                        const currentWidth = isResizable ? DURATION_WIDTHS[duration || 'short'] : null;
                                        const isDragging = dragId === s.id;
                                        const dragIdx = dragId ? sections.findIndex(sec => sec.id === dragId) : -1;
                                        const showGapBefore = dropIndex === index && dragId !== null && dragIdx !== index && dragIdx !== index - 1;
                                        const showGapAfter = index === sections.length - 1 && dropIndex === sections.length && dragId !== null && dragIdx !== sections.length - 1;
                                        return (
                                            <React.Fragment key={s.id}>
                                                <div style={{ width: showGapBefore ? '32px' : '4px', height: '22px', backgroundColor: showGapBefore ? '#10B98130' : 'transparent', borderRadius: '3px', border: showGapBefore ? '1.5px dashed #10B981' : 'none', transition: 'all 0.2s ease', flexShrink: 0 }} />
                                                <div draggable={!resizingId} onDragStart={e => !resizingId && handleDragStart(e, s.id)} onDragOver={e => handleDragOverWithPosition(e, s.id, index)} onDragEnd={handleDragEnd} className="section-pill"
                                                    style={{ position: 'relative', width: isResizable ? currentWidth : 'auto', minWidth: isResizable ? currentWidth : 'auto', padding: '4px 8px', borderRadius: '5px', backgroundColor: cfg.color + '20', border: `1px solid ${cfg.color}`, color: cfg.color, fontSize: '10px', fontWeight: '500', cursor: resizingId ? 'ew-resize' : 'grab', display: 'flex', alignItems: 'center', justifyContent: 'flex-start', flexShrink: 0, opacity: isDragging ? 0.4 : 1, transform: isDragging ? 'scale(0.95)' : 'scale(1)', transition: resizingId ? 'none' : 'transform 0.15s ease, opacity 0.15s ease' }}>
                                                    <span style={{ display: 'flex', alignItems: 'center', gap: '3px', userSelect: 'none' }}>
                                                        <DragIcon size={8} color={cfg.color} style={{ opacity: 0.4 }} />
                                                        {cfg.name}
                                                        {isResizable && <span style={{ fontSize: '8px', opacity: 0.7 }}>{duration === 'medium' ? 'M' : 'S'}</span>}
                                                    </span>
                                                    {isResizable && (
                                                        <div onMouseDown={(e) => handleResizeStart(e, s.id, currentWidth)} draggable={false} style={{ position: 'absolute', right: '-2px', top: '2px', bottom: '2px', width: '6px', cursor: 'ew-resize', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '2px', backgroundColor: resizingId === s.id ? cfg.color + '80' : cfg.color + '50', transition: resizingId === s.id ? 'none' : 'background-color 0.15s' }} title="Drag to resize" />
                                                    )}
                                                </div>
                                                {showGapAfter && <div style={{ width: '32px', height: '22px', marginLeft: '4px', backgroundColor: '#10B98130', borderRadius: '3px', border: '1.5px dashed #10B981', transition: 'all 0.2s ease', flexShrink: 0 }} />}
                                            </React.Fragment>
                                        );
                                    })}
                                    <div style={{ width: '4px', flexShrink: 0 }} />
                                    <button ref={addBtnRef} onClick={toggleAddMenu} style={{ width: '22px', height: '22px', borderRadius: '5px', backgroundColor: '#1e1e1e', border: '1px dashed #3a3a3a', color: '#666', fontSize: '14px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, transition: 'all 0.15s' }} onMouseEnter={e => { e.currentTarget.style.borderColor = '#10B981'; e.currentTarget.style.color = '#10B981'; }} onMouseLeave={e => { e.currentTarget.style.borderColor = '#3a3a3a'; e.currentTarget.style.color = '#666'; }}>+</button>
                                </div>
                                <div style={{ marginTop: '8px', fontSize: '11px', color: '#666' }}>
                                    Use Verse/Chorus/Bridge for lyrics. Intro/Outro/Instrumental are instrumental only. Each line becomes a sentence. Instrumental (inst) is less stable.
                                </div>
                            </Card>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                {sections.map(s => <SectionCard key={s.id} section={s} onUpdate={u => updateSection(s.id, u)} onRemove={() => removeSection(s.id)} />)}
                            </div>
                        </main>

                        {/* Right Sidebar - Activity */}
                        <aside style={{ width: '280px', flexShrink: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden', paddingBottom: '100px' }}>
                            <div style={{ backgroundColor: '#282828', borderRadius: '12px', padding: '16px', border: '1px solid #333', display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
                                <div style={{ fontSize: '14px', fontWeight: '600', color: '#999', marginBottom: '12px', flexShrink: 0 }}>Songs</div>
                                <div {...rightHoverHandlers} style={{ flex: 1, overflowY: 'auto', paddingRight: '8px', ...getScrollStyle(rightHover) }}>
                                    {!currentGenPayload && queue.length === 0 && library.length === 0 ? (
                                        <div style={{ color: '#555', fontSize: '12px', textAlign: 'center', padding: '20px 0' }}>No activity yet</div>
                                    ) : (
                                        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', paddingBottom: '80px' }}>
                                            {/* Filter queue to exclude items that are already generating (race condition fix) */}
                                            {(() => {
                                                const generatingIds = new Set(
                                                    library.filter(item => ['generating', 'processing', 'pending'].includes(item.status)).map(item => item.id)
                                                );
                                                if (currentGenId) generatingIds.add(currentGenId);
                                                const filteredQueue = queue.filter(item => !generatingIds.has(item.id));
                                                return filteredQueue.slice().reverse().map((item, idx) => (
                                                    <div key={`q-${item.id}`} className="activity-item" style={{ display: 'flex', gap: '10px', alignItems: 'center', position: 'relative' }}>
                                                        <button onClick={() => removeFromQueue(item.id).then(loadQueue)} style={{ position: 'absolute', top: '4px', right: '4px', background: 'none', border: 'none', color: '#666', cursor: 'pointer' }}><CloseIcon size={10} /></button>
                                                        <div style={{ width: '44px', height: '44px', borderRadius: '6px', backgroundColor: '#6366F1', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontSize: '16px', fontWeight: '600' }}>{filteredQueue.length - idx}</div>
                                                        <div style={{ flex: 1 }}><div className="text-sm font-medium text-primary truncate">{item.title || 'Untitled'}</div><div className="text-xs text-muted">In queue</div></div>
                                                    </div>
                                                ));
                                            })()}
                                            {currentGenPayload ? (
                                                <div className="activity-item processing" style={{ display: 'flex', gap: '10px', alignItems: 'center', position: 'relative', overflow: 'hidden' }}>
                                                    <div style={{ position: 'absolute', top: 0, left: 0, bottom: 0, width: estimatedTime > 0 && elapsedTime > 0 ? `${Math.min((elapsedTime / estimatedTime) * 100, 100)}%` : '0%', background: 'linear-gradient(90deg, rgba(245, 158, 11, 0.15) 0%, rgba(245, 158, 11, 0.05) 100%)', zIndex: 0, transition: 'width 0.5s ease' }} />
                                                    {/* No stop button - running generations cannot be stopped */}
                                                    <div style={{ width: '44px', height: '44px', borderRadius: '6px', backgroundColor: '#F59E0B', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1 }}><SpinnerIcon size={18} /></div>
                                                    <div style={{ flex: 1, zIndex: 1 }}><div className="text-sm font-medium text-primary truncate">{currentGenPayload.title || 'Untitled'}</div><div className="text-xs text-secondary">{elapsedTime > 0 ? `${formatTime(elapsedTime)}${estimatedTime > 0 ? ` / ~${formatTime(estimatedTime)}` : ''}` : status || 'Starting...'}</div></div>
                                                </div>
                                            ) : generating && queue.length > 0 && (
                                                <div className="activity-item processing" style={{ display: 'flex', gap: '10px', alignItems: 'center', position: 'relative', overflow: 'hidden' }}>
                                                    <div style={{ width: '44px', height: '44px', borderRadius: '6px', backgroundColor: '#F59E0B', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><SpinnerIcon size={18} /></div>
                                                    <div style={{ flex: 1 }}><div className="text-sm font-medium text-primary truncate">Starting next song...</div><div className="text-xs text-secondary">Please wait</div></div>
                                                </div>
                                            )}
                                            {library.filter(item => item.id !== currentGenId && !['generating', 'processing', 'pending'].includes(item.status)).map(item => <SongsPanelItem key={item.id} item={item} audioPlayer={audioPlayer} onDelete={() => deleteGeneration(item.id).then(loadLibrary)} />)}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </aside>
                    </div>
                </div>

                {/* Library View */}
                <div style={{ display: activeTab === 'library' ? 'flex' : 'none', flex: 1, overflow: 'hidden' }}>
                    <div {...libraryHoverHandlers} style={{ maxWidth: '900px', margin: '0 auto', padding: '0 24px 100px 24px', flex: 1, overflowY: 'auto', ...getScrollStyle(libraryHover) }}>
                        <h2 style={{ fontSize: '20px', fontWeight: '600', marginBottom: '20px', color: '#e0e0e0' }}>
                            Your Songs ({library.filter(l => l.status === 'completed').length})
                            {(currentGenPayload || queue.length > 0) && <span style={{ color: '#6366F1', fontWeight: '400', fontSize: '14px', marginLeft: '12px' }}>+ {(currentGenPayload ? 1 : 0) + queue.length} pending</span>}
                        </h2>
                        {library.length === 0 && queue.length === 0 && !currentGenPayload ? (
                            <div style={{ textAlign: 'center', padding: '60px', color: '#666' }}>No songs generated yet. Start creating!</div>
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                {/* Filter queue to exclude items that are already generating (race condition fix) */}
                                {(() => {
                                    const generatingIds = new Set(
                                        library.filter(item => ['generating', 'processing', 'pending'].includes(item.status)).map(item => item.id)
                                    );
                                    if (currentGenId) generatingIds.add(currentGenId);
                                    const filteredQueue = queue.filter(item => !generatingIds.has(item.id));
                                    return filteredQueue.slice().reverse().map((item, idx) => (
                                        <LibraryItem key={`queue-${item.id}`} item={item} isQueued queuePosition={filteredQueue.length - idx} onRemoveFromQueue={() => removeFromQueue(item.id).then(loadQueue)} />
                                    ));
                                })()}
                                {currentGenPayload ? (
                                    <LibraryItem item={currentGenPayload} isGenerating status={status} elapsedTime={elapsedTime} estimatedTime={estimatedTime} />
                                ) : generating && queue.length > 0 && (
                                    <LibraryItem item={{ title: 'Starting next song...', status: 'pending' }} isGenerating status="Please wait" elapsedTime={0} estimatedTime={0} />
                                )}
                                {library.filter(item => item.id !== currentGenId && !['generating', 'processing', 'pending'].includes(item.status)).map(item => (
                                    <LibraryItem
                                        key={item.id}
                                        item={item}
                                        onDelete={() => deleteGeneration(item.id).then(loadLibrary)}
                                        onPlay={audioPlayer.play}
                                        onUpdate={loadLibrary}
                                        isCurrentlyPlaying={audioPlayer.playingId === item.id}
                                        isAudioPlaying={audioPlayer.isPlaying}
                                        playingTrackIdx={audioPlayer.playingTrackIdx}
                                    />
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Add Section Popup */}
            {showAddMenu && (
                <div className="add-section-popup" style={{ left: addMenuPos.x, top: addMenuPos.y, transform: addMenuPos.openUpward ? 'translateY(-100%)' : 'none' }}>
                    {Object.entries(SECTION_TYPES).filter(([key]) => !['intro', 'outro'].includes(key) || !sections.some(s => fromApiType(s.type).base === key)).map(([key, val]) => (
                        <button key={key} onClick={() => addSection(key)} style={{ color: val.color }}>+ {val.name}</button>
                    ))}
                </div>
            )}

            {/* Media Player Footer */}
            <footer style={{ position: 'fixed', bottom: '16px', left: '50%', transform: 'translateX(-50%)', width: 'calc(100% - 64px)', maxWidth: '900px', background: 'linear-gradient(135deg, rgba(25, 25, 25, 0.85) 0%, rgba(35, 35, 35, 0.9) 100%)', backdropFilter: 'blur(24px)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '16px', padding: '12px 20px', display: 'flex', alignItems: 'center', gap: '20px', zIndex: 100 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', minWidth: '200px', maxWidth: '300px' }}>
                    <div style={{ width: '48px', height: '48px', borderRadius: '6px', backgroundColor: audioPlayer.playingItem ? '#10B981' : '#3a3a3a', backgroundImage: audioPlayer.playingItem?.metadata?.cover ? `url(/api/generation/${audioPlayer.playingItem.id}/cover)` : 'none', backgroundSize: 'cover', backgroundPosition: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        {!audioPlayer.playingItem?.metadata?.cover && <MusicNoteIcon size={20} color={audioPlayer.playingItem ? '#fff' : '#666'} />}
                    </div>
                    <div style={{ overflow: 'hidden' }}>
                        <div style={{ fontSize: '13px', fontWeight: '500', color: audioPlayer.playingItem ? '#fff' : '#666', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{audioPlayer.playingItem ? (audioPlayer.playingItem.metadata?.title || 'Untitled') : 'No song selected'}</div>
                        <div style={{ fontSize: '11px', color: '#888' }}>{audioPlayer.playingItem ? ([audioPlayer.playingItem.metadata?.genre, audioPlayer.playingItem.metadata?.mood].filter(Boolean).join(' • ') || 'No tags') : 'Select a song to play'}</div>
                    </div>
                </div>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                        <button onClick={audioPlayer.playPrev} disabled={!audioPlayer.playingItem} style={{ background: 'none', border: 'none', color: audioPlayer.playingItem ? '#888' : '#444', cursor: audioPlayer.playingItem ? 'pointer' : 'not-allowed' }}><SkipBackIcon /></button>
                        <button onClick={() => audioPlayer.playingItem && audioPlayer.play(audioPlayer.playingItem)} disabled={!audioPlayer.playingItem} style={{ background: audioPlayer.playingItem ? '#fff' : '#555', border: 'none', borderRadius: '50%', width: '36px', height: '36px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: audioPlayer.playingItem ? 'pointer' : 'not-allowed' }}>
                            {audioPlayer.isPlaying ? <PauseLargeIcon size={16} color={audioPlayer.playingItem ? '#000' : '#333'} /> : <PlayLargeIcon size={16} color={audioPlayer.playingItem ? '#000' : '#333'} style={{ marginLeft: '2px' }} />}
                        </button>
                        <button onClick={audioPlayer.playNext} disabled={!audioPlayer.playingItem} style={{ background: 'none', border: 'none', color: audioPlayer.playingItem ? '#888' : '#444', cursor: audioPlayer.playingItem ? 'pointer' : 'not-allowed' }}><SkipForwardIcon /></button>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', width: '100%', maxWidth: '600px' }}>
                        <span style={{ fontSize: '11px', color: '#888', minWidth: '35px', textAlign: 'right' }}>{formatTime(audioPlayer.progress)}</span>
                        <div style={{ flex: 1, height: '4px', backgroundColor: '#3a3a3a', borderRadius: '2px', cursor: audioPlayer.playingItem ? 'pointer' : 'default', position: 'relative' }} onClick={(e) => { if (!audioPlayer.playingItem) return; const rect = e.currentTarget.getBoundingClientRect(); audioPlayer.seek((e.clientX - rect.left) / rect.width * audioPlayer.duration); }}>
                            <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${audioPlayer.duration ? (audioPlayer.progress / audioPlayer.duration) * 100 : 0}%`, backgroundColor: '#10B981', borderRadius: '2px' }} />
                        </div>
                        <span style={{ fontSize: '11px', color: '#888', minWidth: '35px' }}>{formatTime(audioPlayer.duration)}</span>
                    </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: '120px' }}>
                    <button onClick={() => audioPlayer.setVolume(audioPlayer.volume > 0 ? 0 : 1)} style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer' }}>{audioPlayer.volume === 0 ? <VolumeMuteIcon /> : <VolumeFullIcon />}</button>
                    <input type="range" min="0" max="1" step="0.01" value={audioPlayer.volume} onChange={(e) => audioPlayer.setVolume(parseFloat(e.target.value))} style={{ width: '80px', accentColor: '#10B981' }} />
                </div>
            </footer>

            {/* Model Manager Modal */}
            {showModelManager && (
                <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0, 0, 0, 0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }} onClick={() => setShowModelManager(false)}>
                    <div style={{ backgroundColor: '#1a1a1a', borderRadius: '16px', padding: '24px', width: '500px', maxWidth: '90vw', maxHeight: '80vh', overflow: 'auto', border: '1px solid #333' }} onClick={e => e.stopPropagation()}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                            <h2 style={{ margin: 0, fontSize: '18px', fontWeight: '600', color: '#fff' }}>Model Manager</h2>
                            <button onClick={() => setShowModelManager(false)} style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: '20px' }}>×</button>
                        </div>
                        {gpuInfo?.gpu && <div style={{ backgroundColor: '#252525', borderRadius: '8px', padding: '12px', marginBottom: '16px', fontSize: '12px' }}><div style={{ color: '#fff', fontWeight: '500', marginBottom: '4px' }}>{gpuInfo.gpu.name}</div><div style={{ color: '#888' }}>{gpuInfo.gpu.free_gb}GB available / {gpuInfo.gpu.total_gb}GB total</div></div>}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            {modelState.allModels.map(m => {
                                const downloadDetails = formatDownloadDetails(m);
                                return (
                                    <div key={m.id} style={{ backgroundColor: '#252525', borderRadius: '12px', padding: '16px', border: m.status === 'ready' ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid #333' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                                            <div><div style={{ fontSize: '14px', fontWeight: '500', color: '#fff', marginBottom: '4px' }}>{m.name}{m.id === modelState.recommendedModel && <span style={{ marginLeft: '8px', fontSize: '10px', backgroundColor: 'rgba(16, 185, 129, 0.2)', color: '#10B981', padding: '2px 6px', borderRadius: '4px' }}>Recommended</span>}</div><div style={{ fontSize: '12px', color: '#888' }}>{m.description}</div></div>
                                            <div style={{ textAlign: 'right', fontSize: '11px', color: '#666' }}><div>{m.size_gb}GB</div><div>{m.vram_required}GB VRAM</div></div>
                                        </div>
                                        {m.status === 'ready' && <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}><span style={{ fontSize: '12px', color: '#10B981' }}><CheckIcon /> Ready</span><button onClick={() => modelState.deleteModel(m.id)} style={{ padding: '6px 12px', fontSize: '11px', backgroundColor: 'transparent', color: '#EF4444', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '6px', cursor: 'pointer' }}>Delete</button></div>}
                                        {m.status === 'downloading' && (() => {
                                            const stageLabel = m.download_stage === 'runtime' ? 'Runtime assets' : 'Model files';
                                            const isIndeterminate = !m.progress || m.progress <= 0;
                                            return (
                                                <div>
                                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                                                        <span style={{ fontSize: '12px', color: '#F59E0B' }}>Downloading {stageLabel}... {m.progress || 0}%</span>
                                                        <button onClick={() => modelState.cancelDownload(m.id)} style={{ padding: '4px 10px', fontSize: '11px', backgroundColor: 'transparent', color: '#EF4444', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '6px', cursor: 'pointer' }}>Cancel</button>
                                                    </div>
                                                    <div className={isIndeterminate ? 'progress-bar-indeterminate' : ''} style={{ height: '6px', backgroundColor: '#333', borderRadius: '3px', overflow: 'hidden' }}>
                                                        {!isIndeterminate && <div style={{ width: `${m.progress || 0}%`, height: '100%', backgroundColor: '#F59E0B', transition: 'width 0.3s' }} />}
                                                    </div>
                                                    {downloadDetails && <div style={{ marginTop: '8px', fontSize: '11px', color: '#C0841A' }}>{downloadDetails}</div>}
                                                </div>
                                            );
                                        })()}
                                        {m.status === 'coming_soon' && (
                                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                <span style={{ fontSize: '12px', color: '#F59E0B' }}>Coming soon</span>
                                                <button disabled style={{ padding: '6px 12px', fontSize: '11px', backgroundColor: 'transparent', color: '#555', border: '1px solid #333', borderRadius: '6px', cursor: 'not-allowed' }}>Unavailable</button>
                                            </div>
                                        )}
                                        {m.status === 'not_downloaded' && <button onClick={() => modelState.startDownload(m.id)} style={{ width: '100%', padding: '10px', fontSize: '12px', backgroundColor: '#10B981', color: '#fff', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: '500' }}>Download ({m.size_gb}GB)</button>}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

// Render the app
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
