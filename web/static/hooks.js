// SongGeneration Studio - Custom Hooks

// ============ Standalone Time Estimation Functions ============
// Used when fresh timing stats are fetched and we need to calculate estimate immediately

// Calculate estimate from pre-computed values (more reliable for queue items)
var calculateEstimateFromValues = (timingStats, model, numSections, totalLyrics, hasReference = false) => {
    const hasLyrics = totalLyrics > 0;

    // Get model defaults
    const baseTime = MODEL_BASE_TIMES[model] || 180;
    const maxAdditional = MODEL_MAX_ADDITIONAL[model] || 180;

    // Calculate complexity factor (0 to 1)
    const sectionFactor = Math.min(1, Math.max(0, (numSections - 1) / 9));
    const lyricsFactor = Math.min(1, totalLyrics / 2000);
    const complexity = (sectionFactor * 0.4) + (lyricsFactor * 0.6);

    // Calculate default estimate
    let defaultEstimate = baseTime + Math.round(complexity * maxAdditional);
    if (hasReference) defaultEstimate += 30;

    // Determine complexity bucket
    const getComplexityBucket = (sections, lyrics) => {
        const sectionScore = sections <= 3 ? 0 : (sections <= 6 ? 1 : 2);
        let lyricsScore = 0;
        if (lyrics > 0 && lyrics <= 500) lyricsScore = 1;
        else if (lyrics > 500 && lyrics <= 1500) lyricsScore = 2;
        else if (lyrics > 1500) lyricsScore = 3;
        const total = sectionScore + lyricsScore;
        if (total <= 1) return "low";
        if (total <= 3) return "medium";
        if (total <= 4) return "high";
        return "very_high";
    };

    const bucket = getComplexityBucket(numSections, totalLyrics);

    // If we have history, blend with learned values
    if (timingStats?.has_history && timingStats.models?.[model]) {
        const modelStats = timingStats.models[model];
        const recordCount = modelStats.count || 0;
        let learnedEstimate = null;

        if (modelStats.by_bucket?.[bucket]) {
            learnedEstimate = modelStats.by_bucket[bucket];
        } else if (hasLyrics && modelStats.avg_with_lyrics) {
            learnedEstimate = modelStats.avg_with_lyrics;
        } else if (!hasLyrics && modelStats.avg_without_lyrics) {
            learnedEstimate = modelStats.avg_without_lyrics;
        } else if (modelStats.avg_time) {
            learnedEstimate = modelStats.avg_time;
        }

        if (learnedEstimate) {
            if (hasReference && modelStats.avg_with_reference && modelStats.avg_without_reference) {
                const refRatio = modelStats.avg_with_reference / modelStats.avg_without_reference;
                learnedEstimate = Math.round(learnedEstimate * refRatio);
            }
            const confidence = Math.min(0.95, 0.5 + (recordCount * 0.09));
            const blendedEstimate = Math.round(
                (learnedEstimate * confidence) + (defaultEstimate * (1 - confidence))
            );
            return Math.max(60, blendedEstimate);
        }
    }

    return Math.max(60, defaultEstimate);
};

// Calculate estimate from sections array (wrapper for backward compatibility)
var calculateEstimate = (timingStats, model, sectionsList, hasReference = false) => {
    const numSections = sectionsList.length;
    const totalLyrics = sectionsList.reduce((acc, s) => acc + (s.lyrics || '').length, 0);
    return calculateEstimateFromValues(timingStats, model, numSections, totalLyrics, hasReference);
};

// ============ Hover Hook ============
var useHover = () => {
    const [isHovered, setIsHovered] = useState(false);
    const handlers = {
        onMouseEnter: () => setIsHovered(true),
        onMouseLeave: () => setIsHovered(false),
    };
    return [isHovered, handlers];
};

// ============ Models Hook ============
var useModels = () => {
    const [models, setModels] = useState([]);
    const [allModels, setAllModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('songgeneration_base');
    const [hasReadyModel, setHasReadyModel] = useState(false);
    const [recommendedModel, setRecommendedModel] = useState(null);
    const [isInitializing, setIsInitializing] = useState(true);
    const [downloadPolling, setDownloadPolling] = useState(false);
    const [autoDownloadStarting, setAutoDownloadStarting] = useState(false);
    const autoDownloadTriggeredRef = useRef(false);

    const loadModels = useCallback(async (triggerAutoDownload = false) => {
        try {
            const r = await fetch('/api/models');
            if (!r.ok) throw new Error(`Failed to load models: ${r.status}`);
            const d = await r.json();
            
            const all = d.models || [];
            const ready = d.ready_models || all.filter(m => m.status === 'ready');

            setAllModels(all);
            setModels(ready);
            setHasReadyModel(d.has_ready_model);
            setRecommendedModel(d.recommended);
            if (d.default) setSelectedModel(d.default);

            const hasDownloading = all.some(m => m.status === 'downloading');
            setDownloadPolling(hasDownloading);

            // Auto-download recommended model on first launch
            if (triggerAutoDownload && !d.has_ready_model && d.recommended && 
                !autoDownloadTriggeredRef.current && !hasDownloading) {
                autoDownloadTriggeredRef.current = true;
                setAutoDownloadStarting(true);
                try {
                    const downloadRes = await fetch(`/api/models/${d.recommended}/download`, { method: 'POST' });
                    if (downloadRes.ok) {
                        setDownloadPolling(true);
                        loadModels();
                    }
                } catch (e) {
                    console.error('[AUTO-DOWNLOAD] Error:', e);
                    autoDownloadTriggeredRef.current = false;
                }
                setAutoDownloadStarting(false);
            }
            setIsInitializing(false);
        } catch (e) {
            console.error('[MODELS] Error:', e);
            setIsInitializing(false);
        }
    }, []);

    const startDownload = useCallback(async (modelId) => {
        try {
            const r = await fetch(`/api/models/${modelId}/download`, { method: 'POST' });
            if (r.ok) {
                setDownloadPolling(true);
                loadModels();
            }
        } catch (e) { console.error(e); }
    }, [loadModels]);

    const cancelDownload = useCallback(async (modelId) => {
        try {
            await fetch(`/api/models/${modelId}/download`, { method: 'DELETE' });
            setAllModels(prev => {
                const updated = prev.map(m => 
                    m.id === modelId ? { ...m, status: 'not_downloaded', progress: 0 } : m
                );
                if (!updated.some(m => m.status === 'downloading')) {
                    setDownloadPolling(false);
                }
                return updated;
            });
            await loadModels();
        } catch (e) { console.error(e); }
    }, [loadModels]);

    const deleteModelHandler = useCallback(async (modelId) => {
        if (!confirm(`Delete model ${modelId}? You'll need to download it again to use it.`)) return;
        const r = await fetch(`/api/models/${modelId}`, { method: 'DELETE' });
        if (r.ok) loadModels();
    }, [loadModels]);

    // Poll for download progress
    useEffect(() => {
        if (!downloadPolling) return;
        const interval = setInterval(loadModels, 5000);
        return () => clearInterval(interval);
    }, [downloadPolling, loadModels]);

    return {
        models, allModels, selectedModel, setSelectedModel,
        hasReadyModel, recommendedModel, isInitializing,
        downloadPolling, autoDownloadStarting,
        loadModels, startDownload, cancelDownload, deleteModel: deleteModelHandler,
        setAllModels, setModels, setHasReadyModel, setDownloadPolling,
    };
};

// ============ Audio Player Hook ============
var useAudioPlayer = (library) => {
    const [playingId, setPlayingId] = useState(null);
    const [playingItem, setPlayingItem] = useState(null);
    const [playingTrackIdx, setPlayingTrackIdx] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [progress, setProgress] = useState(0);
    const [duration, setDuration] = useState(0);
    const [volume, setVolume] = useState(1);
    const audioRef = useRef(null);
    const playNextRef = useRef(null);

    const seek = useCallback((time) => {
        if (audioRef.current) {
            audioRef.current.currentTime = time;
            setProgress(time);
        }
    }, []);

    const setVolumeHandler = useCallback((vol) => {
        setVolume(vol);
        if (audioRef.current) audioRef.current.volume = vol;
    }, []);

    const play = useCallback((item, trackIdx = 0) => {
        if (!item.output_file && (!item.output_files || item.output_files.length === 0)) return;

        // Toggle play/pause for same item and same track
        if (playingId === item.id && playingTrackIdx === trackIdx && audioRef.current) {
            if (audioRef.current.paused) {
                audioRef.current.play().then(() => setIsPlaying(true));
            } else {
                audioRef.current.pause();
                setIsPlaying(false);
            }
            return;
        }

        // Stop current audio
        if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current.src = '';
            audioRef.current = null;
        }

        setPlayingId(item.id);
        setPlayingItem(item);
        setPlayingTrackIdx(trackIdx);
        setProgress(0);
        setDuration(0);
        setIsPlaying(true);

        const audio = new Audio();
        audio.preload = 'auto';
        audio.volume = volume;
        audio.onended = () => {
            setIsPlaying(false);
            setProgress(0);
            // Use ref to get current playNext function
            if (playNextRef.current) playNextRef.current();
        };
        audio.onerror = () => setIsPlaying(false);
        audio.ontimeupdate = () => setProgress(audio.currentTime);
        audio.onloadedmetadata = () => setDuration(audio.duration);
        audio.src = `/api/audio/${item.id}/${trackIdx}`;
        audioRef.current = audio;
        audio.play().then(() => setIsPlaying(true)).catch(() => setIsPlaying(false));
    }, [playingId, playingTrackIdx, volume]);

    const playNext = useCallback(() => {
        if (!playingItem) return;
        const songs = library.filter(l => l.status === 'completed' && (l.output_file || l.output_files?.length > 0));
        const idx = songs.findIndex(l => l.id === playingItem.id);
        if (idx >= 0 && idx < songs.length - 1) play(songs[idx + 1]);
    }, [playingItem, library, play]);

    const playPrev = useCallback(() => {
        if (!playingItem) return;
        const songs = library.filter(l => l.status === 'completed' && (l.output_file || l.output_files?.length > 0));
        const idx = songs.findIndex(l => l.id === playingItem.id);
        if (idx > 0) play(songs[idx - 1]);
    }, [playingItem, library, play]);

    // Keep playNextRef updated with current playNext function
    useEffect(() => {
        playNextRef.current = playNext;
    }, [playNext]);

    // Update playingItem when library changes
    useEffect(() => {
        if (playingItem) {
            const updated = library.find(item => item.id === playingItem.id);
            if (updated) setPlayingItem(updated);
        }
    }, [library, playingItem]);

    return {
        playingId, playingItem, playingTrackIdx, isPlaying, progress, duration, volume,
        play, seek, setVolume: setVolumeHandler, playNext, playPrev,
    };
};

// ============ Time Estimation Hook ============
//
// Algorithm:
// 1. Start with model default (Base: 3:00, Base Full: 4:00, Large: 6:00)
// 2. Add complexity factor based on lyrics length and section count (up to max additional)
// 3. If we have history, blend the default estimate with learned average using EMA
// 4. The more history we have, the more we trust the learned values
//
var useTimeEstimation = (timingStats) => {
    return useCallback((model, sectionsList, hasReference = false) => {
        const numSections = sectionsList.length;
        const totalLyrics = sectionsList.reduce((acc, s) => acc + (s.lyrics || '').length, 0);
        const hasLyrics = totalLyrics > 0;

        // Get model defaults
        const baseTime = MODEL_BASE_TIMES[model] || 180;
        const maxAdditional = MODEL_MAX_ADDITIONAL[model] || 180;

        // Calculate complexity factor (0 to 1)
        // Sections: normalized 0-1 based on 1-10 sections
        const sectionFactor = Math.min(1, Math.max(0, (numSections - 1) / 9));
        // Lyrics: normalized 0-1 based on 0-2000 characters
        const lyricsFactor = Math.min(1, totalLyrics / 2000);
        // Combined complexity (weighted average: lyrics matter more)
        const complexity = (sectionFactor * 0.4) + (lyricsFactor * 0.6);

        // Calculate default estimate (before learning)
        let defaultEstimate = baseTime + Math.round(complexity * maxAdditional);

        // Reference audio adds ~30 seconds
        if (hasReference) {
            defaultEstimate += 30;
        }

        // Determine complexity bucket (must match backend logic)
        const getComplexityBucket = (sections, lyrics) => {
            const sectionScore = sections <= 3 ? 0 : (sections <= 6 ? 1 : 2);
            let lyricsScore = 0;
            if (lyrics > 0 && lyrics <= 500) lyricsScore = 1;
            else if (lyrics > 500 && lyrics <= 1500) lyricsScore = 2;
            else if (lyrics > 1500) lyricsScore = 3;
            const total = sectionScore + lyricsScore;
            if (total <= 1) return "low";
            if (total <= 3) return "medium";
            if (total <= 4) return "high";
            return "very_high";
        };

        const bucket = getComplexityBucket(numSections, totalLyrics);

        // If we have history for this model, blend with learned values
        if (timingStats?.has_history && timingStats.models?.[model]) {
            const modelStats = timingStats.models[model];
            const recordCount = modelStats.count || 0;

            // Get learned estimate from matching bucket, or fall back to general average
            let learnedEstimate = null;

            // Try exact bucket match first
            if (modelStats.by_bucket?.[bucket]) {
                learnedEstimate = modelStats.by_bucket[bucket];
            }
            // Fall back to lyrics-based average
            else if (hasLyrics && modelStats.avg_with_lyrics) {
                learnedEstimate = modelStats.avg_with_lyrics;
            } else if (!hasLyrics && modelStats.avg_without_lyrics) {
                learnedEstimate = modelStats.avg_without_lyrics;
            }
            // Fall back to general average
            else if (modelStats.avg_time) {
                learnedEstimate = modelStats.avg_time;
            }

            if (learnedEstimate) {
                // Adjust learned estimate for reference if we have data
                if (hasReference && modelStats.avg_with_reference && modelStats.avg_without_reference) {
                    const refRatio = modelStats.avg_with_reference / modelStats.avg_without_reference;
                    learnedEstimate = Math.round(learnedEstimate * refRatio);
                }

                // Blend default with learned based on confidence (more records = more trust)
                // After 1 record: 50% learned, after 5 records: 80% learned, after 10+: 95% learned
                const confidence = Math.min(0.95, 0.5 + (recordCount * 0.09));
                const blendedEstimate = Math.round(
                    (learnedEstimate * confidence) + (defaultEstimate * (1 - confidence))
                );

                return Math.max(60, blendedEstimate);
            }
        }

        // No history - use default estimate
        return Math.max(60, defaultEstimate);
    }, [timingStats]);
};

