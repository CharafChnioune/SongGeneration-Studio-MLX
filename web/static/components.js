// SongGeneration Studio - UI Components
// Dependencies: constants.js, icons.js (loaded before this file)

// Custom dark audio player component
var DarkAudioPlayer = ({ src }) => {
    const audioRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [volume, setVolume] = useState(1);

    const togglePlay = () => {
        if (audioRef.current) {
            if (isPlaying) {
                audioRef.current.pause();
            } else {
                audioRef.current.play();
            }
            setIsPlaying(!isPlaying);
        }
    };

    const handleTimeUpdate = () => {
        if (audioRef.current) {
            setCurrentTime(audioRef.current.currentTime);
        }
    };

    const handleLoadedMetadata = () => {
        if (audioRef.current) {
            const dur = audioRef.current.duration;
            // Only set if valid (not NaN or Infinity)
            if (dur && !isNaN(dur) && isFinite(dur)) {
                setDuration(dur);
            }
        }
    };

    const handleSeek = (e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        if (audioRef.current) {
            audioRef.current.currentTime = percent * duration;
        }
    };

    const handleEnded = () => {
        setIsPlaying(false);
        setCurrentTime(0);
    };

    const progress = (duration > 0 && !isNaN(duration)) ? (currentTime / duration) * 100 : 0;

    return (
        <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            backgroundColor: '#1a1a1a',
            borderRadius: '8px',
            padding: '10px 14px',
            flex: 1,
        }}>
            <audio
                ref={audioRef}
                src={src}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onEnded={handleEnded}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
            />

            {/* Play/Pause button */}
            <button
                onClick={togglePlay}
                style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '50%',
                    border: 'none',
                    backgroundColor: '#10B981',
                    color: '#fff',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                }}
            >
                {isPlaying ? <PauseIcon /> : <PlayIcon />}
            </button>

            {/* Time */}
            <span style={{ fontSize: '12px', color: '#888', minWidth: '70px' }}>
                {formatTime(currentTime)} / {formatTime(duration)}
            </span>

            {/* Progress bar */}
            <div
                onClick={handleSeek}
                style={{
                    flex: 1,
                    height: '6px',
                    backgroundColor: '#333',
                    borderRadius: '3px',
                    cursor: 'pointer',
                    position: 'relative',
                }}
            >
                <div style={{
                    width: `${progress}%`,
                    height: '100%',
                    backgroundColor: '#10B981',
                    borderRadius: '3px',
                    transition: 'width 0.1s',
                }} />
            </div>

            {/* Volume */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <VolumeIcon />
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={volume}
                    onChange={(e) => {
                        const v = parseFloat(e.target.value);
                        setVolume(v);
                        if (audioRef.current) audioRef.current.volume = v;
                    }}
                    style={{ width: '60px', height: '4px' }}
                />
            </div>
        </div>
    );
};

// Audio Trimmer Component with Waveform Visualization
var AudioTrimmer = ({ onAccept, onClear, onFileLoad }) => {
    const [file, setFile] = useState(null);
    const [clipDuration, setClipDuration] = useState(10);
    const [regionStart, setRegionStart] = useState(0);
    const [totalDuration, setTotalDuration] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [isAccepted, setIsAccepted] = useState(false);
    const [isExpanded, setIsExpanded] = useState(false);
    const [waveformPeaks, setWaveformPeaks] = useState([]);
    const [isDragging, setIsDragging] = useState(false);
    const [error, setError] = useState(null);

    const waveformRef = useRef(null);
    const wavesurferRef = useRef(null);
    const fileInputRef = useRef(null);
    const audioContextRef = useRef(null);
    const audioBufferRef = useRef(null);
    const playIntervalRef = useRef(null);

    // Handle drag/click on waveform to move selection
    const handleWaveformInteraction = (e, containerRef, padding = 12) => {
        if (totalDuration <= 0 || isLoading) return;
        const rect = containerRef.getBoundingClientRect();
        const clickX = e.clientX - rect.left - padding;
        const containerWidth = rect.width - (padding * 2);
        const clickPercent = Math.max(0, Math.min(1, clickX / containerWidth));
        const clickTime = clickPercent * totalDuration;
        const newStart = Math.max(0, Math.min(totalDuration - clipDuration, clickTime - clipDuration / 2));
        setRegionStart(newStart);
    };

    const handleMouseDown = (e, containerRef, padding) => {
        setIsDragging(true);
        handleWaveformInteraction(e, containerRef, padding);
    };

    const handleMouseMove = (e, containerRef, padding) => {
        if (!isDragging) return;
        handleWaveformInteraction(e, containerRef, padding);
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    // Global mouse up listener to stop dragging
    useEffect(() => {
        const handleGlobalMouseUp = () => setIsDragging(false);
        window.addEventListener('mouseup', handleGlobalMouseUp);
        return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
    }, []);

    // Initialize WaveSurfer when file is loaded
    useEffect(() => {
        if (!file || !waveformRef.current) return;

        setIsLoading(true);
        setError(null);

        // Clean up previous instance
        if (wavesurferRef.current) {
            wavesurferRef.current.destroy();
        }

        // Create WaveSurfer instance
        const ws = WaveSurfer.create({
            container: waveformRef.current,
            waveColor: '#4a4a4a',
            progressColor: '#4a4a4a',
            cursorColor: 'transparent',
            cursorWidth: 0,
            height: 80,
            barWidth: 2,
            barGap: 1,
            barRadius: 2,
            normalize: true,
            backend: 'WebAudio',
            interact: false,
        });

        wavesurferRef.current = ws;

        // Load audio file
        const url = URL.createObjectURL(file);
        ws.load(url);

        ws.on('ready', () => {
            const duration = ws.getDuration();
            setTotalDuration(duration);

            const initialClipDuration = Math.min(clipDuration, duration);
            setClipDuration(initialClipDuration);
            setRegionStart(0);

            // Use WaveSurfer's decoded data directly (no need to decode again!)
            const decodedBuffer = ws.getDecodedData();
            if (decodedBuffer) {
                audioBufferRef.current = decodedBuffer;

                // Extract peaks for the expanded view
                const numBars = 200;
                const channelData = decodedBuffer.getChannelData(0);
                const samplesPerBar = Math.floor(channelData.length / numBars);
                const peaks = [];
                for (let i = 0; i < numBars; i++) {
                    let max = 0;
                    const start = i * samplesPerBar;
                    for (let j = 0; j < samplesPerBar; j++) {
                        const val = Math.abs(channelData[start + j] || 0);
                        if (val > max) max = val;
                    }
                    peaks.push(max);
                }
                const maxPeak = Math.max(...peaks, 0.01);
                const normalizedPeaks = peaks.map(p => (p / maxPeak) * 100);
                setWaveformPeaks(normalizedPeaks);
            }

            // Only set loading to false after everything is ready
            setIsLoading(false);
        });

        ws.on('error', (err) => {
            setIsLoading(false);
            setError('Failed to load audio file');
            console.error('WaveSurfer error:', err);
        });

        ws.on('finish', () => {
            setIsPlaying(false);
        });

        return () => {
            URL.revokeObjectURL(url);
            if (wavesurferRef.current) {
                wavesurferRef.current.destroy();
                wavesurferRef.current = null;
            }
        };
    }, [file]);

    const handleFileSelect = async (e) => {
        const selectedFile = e.target.files?.[0];
        if (!selectedFile) return;

        const validTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/flac', 'audio/ogg'];
        if (!validTypes.includes(selectedFile.type) && !selectedFile.name.match(/\.(wav|mp3|flac|ogg)$/i)) {
            setError('Please select a valid audio file (WAV, MP3, FLAC, or OGG)');
            return;
        }

        setFile(selectedFile);
        setIsAccepted(false);
        setRegionStart(0);
        if (onFileLoad) onFileLoad(true);
    };

    const stopPlayback = () => {
        if (playIntervalRef.current) {
            clearInterval(playIntervalRef.current);
            playIntervalRef.current = null;
        }
        if (wavesurferRef.current) {
            wavesurferRef.current.pause();
        }
        setIsPlaying(false);
    };

    const handlePreview = () => {
        if (!wavesurferRef.current) return;

        if (isPlaying) {
            stopPlayback();
        } else {
            if (playIntervalRef.current) {
                clearInterval(playIntervalRef.current);
            }

            wavesurferRef.current.setTime(regionStart);
            wavesurferRef.current.play();
            setIsPlaying(true);

            playIntervalRef.current = setInterval(() => {
                if (wavesurferRef.current) {
                    const currentTime = wavesurferRef.current.getCurrentTime();
                    if (currentTime >= regionStart + clipDuration || !wavesurferRef.current.isPlaying()) {
                        stopPlayback();
                    }
                }
            }, 50);
        }
    };

    const audioBufferToWav = (buffer) => {
        const numChannels = buffer.numberOfChannels;
        const sampleRate = buffer.sampleRate;
        const format = 3;
        const bitDepth = 32;

        const bytesPerSample = bitDepth / 8;
        const blockAlign = numChannels * bytesPerSample;
        const byteRate = sampleRate * blockAlign;
        const dataSize = buffer.length * blockAlign;
        const headerSize = 44;
        const totalSize = headerSize + dataSize;

        const arrayBuffer = new ArrayBuffer(totalSize);
        const view = new DataView(arrayBuffer);

        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, totalSize - 8, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(36, 'data');
        view.setUint32(40, dataSize, true);

        const channels = [];
        for (let ch = 0; ch < numChannels; ch++) {
            channels.push(buffer.getChannelData(ch));
        }

        let offset = 44;
        for (let i = 0; i < buffer.length; i++) {
            for (let ch = 0; ch < numChannels; ch++) {
                view.setFloat32(offset, channels[ch][i], true);
                offset += 4;
            }
        }

        return new Blob([arrayBuffer], { type: 'audio/wav' });
    };

    const handleAccept = async () => {
        stopPlayback();

        if (!file) {
            setError('No file selected');
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            // Send original file to server for high-quality trimming with ffmpeg
            const formData = new FormData();
            formData.append('file', file);
            formData.append('trim_start', regionStart.toString());
            formData.append('trim_duration', clipDuration.toString());

            const response = await fetch('/api/upload-and-trim-reference', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(await response.text());
            }

            const data = await response.json();
            setIsAccepted(true);
            setIsLoading(false);

            onAccept({
                id: data.id,
                fileName: file.name,
                clipStart: regionStart,
                clipDuration: clipDuration
            });
        } catch (err) {
            setError(err.message || 'Failed to process audio');
            setIsLoading(false);
        }
    };

    const handleClear = () => {
        if (wavesurferRef.current) {
            wavesurferRef.current.destroy();
            wavesurferRef.current = null;
        }
        setFile(null);
        setIsAccepted(false);
        setRegionStart(0);
        setTotalDuration(0);
        setError(null);
        audioBufferRef.current = null;
        if (onFileLoad) onFileLoad(false);
        onClear();
    };

    const handleDurationChange = (newDuration) => {
        const maxDuration = Math.min(newDuration, totalDuration);
        setClipDuration(maxDuration);

        if (regionStart + maxDuration > totalDuration) {
            setRegionStart(Math.max(0, totalDuration - maxDuration));
        }
    };

    // Render upload state
    if (!file) {
        return (
            <div>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*"
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                />
                <button
                    onClick={() => fileInputRef.current?.click()}
                    style={{
                        width: '100%',
                        padding: '20px',
                        borderRadius: '10px',
                        border: '2px dashed #3a3a3a',
                        backgroundColor: 'transparent',
                        color: '#666',
                        fontSize: '13px',
                        cursor: 'pointer',
                        transition: 'border-color 0.2s',
                    }}
                    onMouseOver={e => e.target.style.borderColor = '#10B981'}
                    onMouseOut={e => e.target.style.borderColor = '#3a3a3a'}
                >
                    Click to upload reference audio
                </button>
                {error && <div style={{ color: '#EF4444', fontSize: '12px', marginTop: '8px' }}>{error}</div>}
            </div>
        );
    }

    // Render accepted state
    if (isAccepted) {
        return (
            <div style={{
                backgroundColor: '#1e1e1e',
                borderRadius: '10px',
                padding: '14px',
                border: '1px solid #10B981',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span style={{ fontSize: '18px' }}>&#x2713;</span>
                        <div>
                            <div style={{ fontSize: '13px', color: '#e0e0e0' }}>{file.name}</div>
                            <div style={{ fontSize: '11px', color: '#10B981' }}>
                                {formatTime(regionStart)} - {formatTime(regionStart + clipDuration)} ({clipDuration}s clip)
                            </div>
                        </div>
                    </div>
                    <button
                        onClick={handleClear}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#666',
                            cursor: 'pointer',
                            fontSize: '16px',
                            padding: '4px 8px',
                        }}
                    >x</button>
                </div>
            </div>
        );
    }

    // Render trimmer UI
    return (
        <div>
            {/* File info */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                backgroundColor: '#1e1e1e',
                padding: '10px 14px',
                borderRadius: '10px',
                marginBottom: '12px',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontSize: '16px' }}>&#x1F3B5;</span>
                    <span style={{ fontSize: '13px', color: '#e0e0e0', maxWidth: '180px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {file.name}
                    </span>
                </div>
                <button
                    onClick={handleClear}
                    style={{
                        background: 'transparent',
                        border: 'none',
                        color: '#666',
                        cursor: 'pointer',
                        fontSize: '16px',
                    }}
                >x</button>
            </div>

            {/* Duration input */}
            <div style={{ marginBottom: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <span style={{ fontSize: '12px', color: '#666' }}>Clip Duration</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            background: '#1e1e1e',
                            border: '1px solid #3a3a3a',
                            borderRadius: '6px',
                            overflow: 'hidden'
                        }}>
                            <input
                                type="text"
                                value={clipDuration}
                                onChange={e => {
                                    const val = parseInt(e.target.value) || 1;
                                    const max = Math.floor(totalDuration) || 60;
                                    handleDurationChange(Math.min(Math.max(1, val), max));
                                }}
                                style={{
                                    width: '36px',
                                    textAlign: 'center',
                                    background: 'transparent',
                                    border: 'none',
                                    color: '#10B981',
                                    fontSize: '13px',
                                    fontWeight: '600',
                                    outline: 'none',
                                    padding: '6px 4px'
                                }}
                            />
                            <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid #3a3a3a' }}>
                                <button
                                    onClick={() => handleDurationChange(Math.min(clipDuration + 1, Math.floor(totalDuration) || 60))}
                                    style={{
                                        background: 'transparent',
                                        border: 'none',
                                        color: '#888',
                                        cursor: 'pointer',
                                        padding: '2px 6px',
                                        fontSize: '8px',
                                        lineHeight: 1,
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center'
                                    }}
                                >▲</button>
                                <button
                                    onClick={() => handleDurationChange(Math.max(clipDuration - 1, 1))}
                                    style={{
                                        background: 'transparent',
                                        border: 'none',
                                        borderTop: '1px solid #3a3a3a',
                                        color: '#888',
                                        cursor: 'pointer',
                                        padding: '2px 6px',
                                        fontSize: '8px',
                                        lineHeight: 1,
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center'
                                    }}
                                >▼</button>
                            </div>
                        </div>
                        <span style={{ fontSize: '12px', color: '#666' }}>seconds</span>
                    </div>
                </div>
            </div>

            {/* Waveform container */}
            <div style={{ position: 'relative' }}>
                {/* Expand button */}
                {totalDuration > 0 && !isLoading && (
                    <button
                        onClick={() => setIsExpanded(true)}
                        style={{
                            position: 'absolute',
                            top: '8px',
                            right: '8px',
                            zIndex: 5,
                            background: 'rgba(0,0,0,0.5)',
                            border: '1px solid #3a3a3a',
                            borderRadius: '6px',
                            padding: '6px 10px',
                            color: '#888',
                            fontSize: '11px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                            transition: 'all 0.2s',
                        }}
                        onMouseOver={e => { e.currentTarget.style.color = '#10B981'; e.currentTarget.style.borderColor = '#10B981'; }}
                        onMouseOut={e => { e.currentTarget.style.color = '#888'; e.currentTarget.style.borderColor = '#3a3a3a'; }}
                    >
                        <ExpandIcon />
                        Expand
                    </button>
                )}

                <div
                    className="waveform-container"
                    style={{
                        position: 'relative',
                        overflow: 'hidden',
                        cursor: totalDuration > 0 ? (isDragging ? 'grabbing' : 'pointer') : 'default',
                        userSelect: 'none',
                    }}
                    onMouseDown={(e) => handleMouseDown(e, e.currentTarget, 12)}
                    onMouseMove={(e) => handleMouseMove(e, e.currentTarget, 12)}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                >
                    {isLoading && (
                        <div style={{
                            position: 'absolute',
                            top: 0, left: 0, right: 0, bottom: 0,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            backgroundColor: 'rgba(26, 26, 26, 0.8)',
                            borderRadius: '8px',
                            zIndex: 10,
                        }}>
                            <span style={{ color: '#10B981', fontSize: '13px' }}>Loading...</span>
                        </div>
                    )}

                    {/* Waveform wrapper with selection overlay */}
                    <div style={{ position: 'relative' }}>
                        <div ref={waveformRef} />

                        {/* Region overlay visualization */}
                        {totalDuration > 0 && !isLoading && (
                            <div style={{
                                position: 'absolute',
                                top: 0,
                                left: `${(regionStart / totalDuration) * 100}%`,
                                width: `${(clipDuration / totalDuration) * 100}%`,
                                height: '100%',
                                backgroundColor: 'rgba(16, 185, 129, 0.25)',
                                borderLeft: '2px solid #10B981',
                                borderRight: '2px solid #10B981',
                                pointerEvents: 'none',
                                boxSizing: 'border-box',
                            }} />
                        )}
                    </div>
                </div>
            </div>

            {/* Expanded waveform popup */}
            {isExpanded && (
                <div
                    style={{
                        position: 'fixed',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        backgroundColor: 'rgba(0, 0, 0, 0.85)',
                        zIndex: 9999,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        padding: '40px',
                        animation: 'fadeIn 0.2s ease-out',
                    }}
                    onClick={(e) => {
                        if (e.target === e.currentTarget) setIsExpanded(false);
                    }}
                >
                    <div style={{
                        width: '100%',
                        maxWidth: '1200px',
                        backgroundColor: '#1e1e1e',
                        borderRadius: '16px',
                        padding: '24px',
                        animation: 'slideUp 0.3s ease-out',
                    }}>
                        {/* Header */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                            <div>
                                <div style={{ fontSize: '16px', fontWeight: '600', color: '#e0e0e0' }}>{file?.name}</div>
                                <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                                    Click on the waveform to position your {clipDuration}s selection
                                </div>
                            </div>
                            <button
                                onClick={() => setIsExpanded(false)}
                                style={{
                                    background: 'transparent',
                                    border: '1px solid #3a3a3a',
                                    borderRadius: '8px',
                                    padding: '8px 16px',
                                    color: '#888',
                                    fontSize: '13px',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '6px',
                                }}
                            >
                                <CloseIcon size={14} />
                                Close
                            </button>
                        </div>

                        {/* Large waveform */}
                        <div
                            style={{
                                backgroundColor: '#1a1a1a',
                                borderRadius: '12px',
                                padding: '20px',
                                cursor: totalDuration > 0 ? (isDragging ? 'grabbing' : 'pointer') : 'default',
                                position: 'relative',
                                userSelect: 'none',
                            }}
                            onMouseDown={(e) => handleMouseDown(e, e.currentTarget, 20)}
                            onMouseMove={(e) => handleMouseMove(e, e.currentTarget, 20)}
                            onMouseUp={handleMouseUp}
                            onMouseLeave={handleMouseUp}
                        >
                            <div style={{ height: '120px', position: 'relative' }}>
                                <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    height: '100%',
                                    gap: '1px',
                                }}>
                                    {waveformPeaks.map((peak, i) => {
                                        const pos = i / waveformPeaks.length;
                                        const inSelection = pos >= (regionStart / totalDuration) && pos <= ((regionStart + clipDuration) / totalDuration);
                                        return (
                                            <div
                                                key={i}
                                                style={{
                                                    flex: 1,
                                                    height: `${Math.max(5, peak)}%`,
                                                    backgroundColor: inSelection ? '#10B981' : '#4a4a4a',
                                                    borderRadius: '1px',
                                                }}
                                            />
                                        );
                                    })}
                                </div>

                                {/* Selection overlay */}
                                <div style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: `${(regionStart / totalDuration) * 100}%`,
                                    width: `${(clipDuration / totalDuration) * 100}%`,
                                    height: '100%',
                                    backgroundColor: 'rgba(16, 185, 129, 0.15)',
                                    borderLeft: '3px solid #10B981',
                                    borderRight: '3px solid #10B981',
                                    pointerEvents: 'none',
                                    boxSizing: 'border-box',
                                }} />
                            </div>

                            {/* Time markers */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px' }}>
                                <span style={{ fontSize: '11px', color: '#666' }}>0:00</span>
                                <span style={{ fontSize: '11px', color: '#666' }}>{formatTime(totalDuration / 2)}</span>
                                <span style={{ fontSize: '11px', color: '#666' }}>{formatTime(totalDuration)}</span>
                            </div>
                        </div>

                        {/* Selection info and controls */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '20px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <span style={{ fontSize: '13px', color: '#888' }}>Clip Duration:</span>
                                    <div style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        background: '#1e1e1e',
                                        border: '1px solid #3a3a3a',
                                        borderRadius: '6px',
                                        overflow: 'hidden'
                                    }}>
                                        <input
                                            type="text"
                                            value={clipDuration}
                                            onChange={e => {
                                                const val = parseInt(e.target.value) || 1;
                                                const max = Math.floor(totalDuration) || 60;
                                                handleDurationChange(Math.min(Math.max(1, val), max));
                                            }}
                                            style={{
                                                width: '40px',
                                                textAlign: 'center',
                                                background: 'transparent',
                                                border: 'none',
                                                color: '#10B981',
                                                fontSize: '14px',
                                                fontWeight: '600',
                                                outline: 'none',
                                                padding: '6px 4px'
                                            }}
                                        />
                                        <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid #3a3a3a' }}>
                                            <button
                                                onClick={() => handleDurationChange(Math.min(clipDuration + 1, Math.floor(totalDuration) || 60))}
                                                style={{
                                                    background: 'transparent',
                                                    border: 'none',
                                                    color: '#888',
                                                    cursor: 'pointer',
                                                    padding: '2px 6px',
                                                    fontSize: '8px',
                                                    lineHeight: 1,
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center'
                                                }}
                                            >▲</button>
                                            <button
                                                onClick={() => handleDurationChange(Math.max(clipDuration - 1, 1))}
                                                style={{
                                                    background: 'transparent',
                                                    border: 'none',
                                                    borderTop: '1px solid #3a3a3a',
                                                    color: '#888',
                                                    cursor: 'pointer',
                                                    padding: '2px 6px',
                                                    fontSize: '8px',
                                                    lineHeight: 1,
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center'
                                                }}
                                            >▼</button>
                                        </div>
                                    </div>
                                    <span style={{ fontSize: '13px', color: '#888' }}>sec</span>
                                </div>
                                <div style={{ fontSize: '14px', color: '#10B981', fontWeight: '500' }}>
                                    Selection: {formatTime(regionStart)} - {formatTime(regionStart + clipDuration)}
                                </div>
                            </div>
                            <div style={{ display: 'flex', gap: '12px' }}>
                                <button className="btn btn-secondary" onClick={handlePreview} style={{ color: isPlaying ? '#10B981' : '#e0e0e0', padding: '10px 20px' }}>
                                    {isPlaying ? <><PauseIcon size={14} /> Stop</> : <><PlayIcon size={14} /> Preview</>}
                                </button>
                                <button className="btn btn-primary" onClick={() => { setIsExpanded(false); handleAccept(); }} style={{ padding: '10px 24px' }}>
                                    Accept Selection
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Selection info */}
            {totalDuration > 0 && (
                <div style={{ marginTop: '8px', display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '11px', color: '#888' }}>
                        Selection: {formatTime(regionStart)} - {formatTime(regionStart + clipDuration)}
                    </span>
                    <span style={{ fontSize: '11px', color: '#666' }}>
                        Total: {formatTime(totalDuration)}
                    </span>
                </div>
            )}

            {/* Controls */}
            <div className="flex gap-3" style={{ marginTop: '12px' }}>
                <button
                    className="btn btn-secondary"
                    onClick={handlePreview}
                    disabled={isLoading || totalDuration === 0}
                    style={{ flex: 1, color: isPlaying ? '#10B981' : '#e0e0e0', cursor: isLoading ? 'not-allowed' : 'pointer' }}
                >
                    {isPlaying ? <><PauseIcon /> Stop</> : <><PlayIcon /> Preview</>}
                </button>
                <button
                    className="btn btn-primary"
                    onClick={handleAccept}
                    disabled={isLoading || totalDuration === 0}
                    style={{ flex: 1, cursor: isLoading ? 'not-allowed' : 'pointer', opacity: isLoading ? 0.7 : 1 }}
                >
                    {isLoading ? 'Processing...' : 'Accept'}
                </button>
            </div>

            {error && <div style={{ color: '#EF4444', fontSize: '12px', marginTop: '8px' }}>{error}</div>}
        </div>
    );
};

// Multi-select with horizontal scrolling suggestions
var MultiSelectWithScroll = ({ suggestions, selected, onChange, placeholder }) => {
    const [input, setInput] = useState('');
    const [showAll, setShowAll] = useState(false);
    const scrollRef = useRef(null);

    const addTag = (tag) => {
        if (tag && !selected.includes(tag)) onChange([...selected, tag]);
        setInput('');
    };

    const removeTag = (tag) => onChange(selected.filter(t => t !== tag));

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && input.trim()) { e.preventDefault(); addTag(input.trim()); }
        else if (e.key === 'Backspace' && !input && selected.length) removeTag(selected[selected.length - 1]);
    };

    const normalizedInput = input.trim().toLowerCase();
    const filteredSuggestions = suggestions.filter(s => !selected.includes(s) && (!normalizedInput || s.toLowerCase().includes(normalizedInput)));
    const visibleSuggestions = (showAll || normalizedInput) ? filteredSuggestions : filteredSuggestions.slice(0, 24);
    const hasMore = !normalizedInput && filteredSuggestions.length > 24;

    return (
        <div>
            <div className="input-base" style={{
                padding: '10px 12px',
                minHeight: '42px',
                display: 'flex',
                flexWrap: 'wrap',
                gap: '6px',
                alignItems: 'center',
            }}>
                {selected.map(tag => (
                    <span key={tag} style={{
                        backgroundColor: '#10B98130',
                        color: '#10B981',
                        padding: '4px 10px',
                        borderRadius: '6px',
                        fontSize: '12px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        whiteSpace: 'nowrap',
                    }}>
                        {tag}
                        <span style={{ cursor: 'pointer', opacity: 0.7, fontSize: '14px' }} onClick={() => removeTag(tag)}>×</span>
                    </span>
                ))}
                <input
                    type="text"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={selected.length === 0 ? placeholder : 'Type to search or add...'}
                    style={{
                        border: 'none',
                        background: 'transparent',
                        color: '#e0e0e0',
                        fontSize: '13px',
                        outline: 'none',
                        flex: 1,
                        minWidth: '80px',
                    }}
                />
                {input.trim() && (
                    <button
                        onClick={() => addTag(input.trim())}
                        style={{
                            border: '1px solid #3a3a3a',
                            backgroundColor: '#1e1e1e',
                            color: '#10B981',
                            padding: '4px 8px',
                            borderRadius: '6px',
                            fontSize: '11px',
                            cursor: 'pointer',
                            flexShrink: 0,
                        }}
                    >
                        Add
                    </button>
                )}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '8px' }}>
                <div style={{ fontSize: '11px', color: '#666' }}>
                    Suggestions: {filteredSuggestions.length}
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                    {selected.length > 0 && (
                        <button
                            onClick={() => onChange([])}
                            style={{ border: 'none', background: 'none', color: '#888', fontSize: '11px', cursor: 'pointer' }}
                        >
                            Clear
                        </button>
                    )}
                    {hasMore && (
                        <button
                            onClick={() => setShowAll(!showAll)}
                            style={{ border: 'none', background: 'none', color: '#10B981', fontSize: '11px', cursor: 'pointer' }}
                        >
                            {showAll ? 'Show less' : 'Show all'}
                        </button>
                    )}
                </div>
            </div>
            {/* Suggestion tags */}
            <div
                ref={scrollRef}
                style={{
                    display: 'flex',
                    gap: '6px',
                    marginTop: '8px',
                    flexWrap: 'wrap',
                    maxHeight: showAll || normalizedInput ? '220px' : '120px',
                    overflowY: 'auto',
                    paddingBottom: '4px',
                    paddingRight: '4px',
                    scrollbarWidth: 'thin',
                    scrollbarColor: '#3a3a3a transparent',
                }}
            >
                {visibleSuggestions.map(tag => (
                    <button
                        key={tag}
                        onClick={() => addTag(tag)}
                        style={{
                            padding: '5px 12px',
                            borderRadius: '16px',
                            border: '1px solid #3a3a3a',
                            backgroundColor: '#2a2a2a',
                            color: '#999',
                            fontSize: '11px',
                            cursor: 'pointer',
                            whiteSpace: 'nowrap',
                            flexShrink: 0,
                            transition: 'all 0.15s',
                        }}
                        onMouseOver={e => { e.target.style.borderColor = '#10B981'; e.target.style.color = '#10B981'; }}
                        onMouseOut={e => { e.target.style.borderColor = '#3a3a3a'; e.target.style.color = '#999'; }}
                    >
                        + {tag}
                    </button>
                ))}
            </div>
        </div>
    );
};

// Card component
var Card = ({ children }) => (
    <div style={{
        backgroundColor: '#282828',
        borderRadius: '16px',
        padding: '20px',
    }}>
        {children}
    </div>
);

// Card title component
var CardTitle = ({ children }) => (
    <div style={{ fontSize: '13px', fontWeight: '500', color: '#888', marginBottom: '14px' }}>
        {children}
    </div>
);

// Section card component
var SectionCard = ({ section, onUpdate, onRemove, onAiEdit, aiBusy }) => {
    const { base } = fromApiType(section.type);
    const cfg = SECTION_TYPES[base] || { name: base, color: '#888', hasLyrics: true };
    const lineCount = Math.max(3, (section.lyrics || '').split('\n').length + 1);
    const handleAiEdit = () => {
        if (!onAiEdit || aiBusy) return;
        onAiEdit();
    };

    return (
        <div style={{
            backgroundColor: '#282828',
            borderRadius: '12px',
            borderLeft: `4px solid ${cfg.color}`,
            overflow: 'hidden',
        }}>
            <div style={{
                padding: '14px 18px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
            }}>
                <span style={{ fontSize: '15px', fontWeight: '500', color: cfg.color }}>{cfg.name}</span>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    {cfg.hasLyrics && (
                        <button
                            onClick={handleAiEdit}
                            disabled={aiBusy}
                            style={{
                                background: 'transparent',
                                border: '1px solid #3a3a3a',
                                color: aiBusy ? '#666' : '#10B981',
                                cursor: aiBusy ? 'not-allowed' : 'pointer',
                                fontSize: '11px',
                                padding: '4px 8px',
                                borderRadius: '8px',
                                lineHeight: 1,
                            }}
                        >{aiBusy ? 'AI...' : 'AI'}</button>
                    )}
                    <button
                        onClick={onRemove}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#555',
                            cursor: 'pointer',
                            fontSize: '20px',
                            padding: '4px 8px',
                            lineHeight: 1,
                            transition: 'color 0.15s',
                        }}
                        onMouseEnter={e => e.target.style.color = '#888'}
                        onMouseLeave={e => e.target.style.color = '#555'}
                    >x</button>
                </div>
            </div>
            {cfg.hasLyrics && (
                <div style={{ padding: '0 18px 16px 18px' }}>
                    <textarea
                        value={section.lyrics || ''}
                        onChange={e => onUpdate({ lyrics: e.target.value })}
                        placeholder="Enter lyrics..."
                        rows={lineCount}
                        style={{
                            width: '100%',
                            backgroundColor: 'transparent',
                            border: 'none',
                            color: '#999',
                            fontSize: '14px',
                            lineHeight: '1.8',
                            resize: 'none',
                            outline: 'none',
                        }}
                    />
                </div>
            )}
        </div>
    );
};

// Edit Modal component
var EditModal = ({ item, onClose, onSave }) => {
    const [title, setTitle] = useState(item.metadata?.title || item.title || '');
    const [coverPreview, setCoverPreview] = useState(null);
    const [coverFile, setCoverFile] = useState(null);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState(null);
    const [removeCover, setRemoveCover] = useState(false);  // Track if user wants to remove cover
    // Initialize hasCover from metadata if available for immediate display
    const [hasCover, setHasCover] = useState(() => {
        return !!(item.metadata?.cover);
    });
    // Use stable cache key based on cover filename
    const coverCacheKey = item.metadata?.cover || '';
    // Store existing cover URL to maintain it during edits
    const [existingCoverUrl, setExistingCoverUrl] = useState(() => {
        if (item.metadata?.cover) {
            return `/api/generation/${item.id}/cover?v=${encodeURIComponent(item.metadata.cover)}`;
        }
        return null;
    });
    const fileInputRef = useRef(null);
    // No need to probe server - trust metadata as source of truth

    const handleFileSelect = (e) => {
        const file = e.target.files[0];
        if (file) {
            setCoverFile(file);
            setRemoveCover(false);  // Reset remove flag when new file selected
            const reader = new FileReader();
            reader.onload = (e) => setCoverPreview(e.target.result);
            reader.readAsDataURL(file);
        }
    };

    const handleRemoveCover = () => {
        setRemoveCover(true);
        setCoverFile(null);
        setCoverPreview(null);
    };

    const handleSave = async () => {
        setSaving(true);
        setError(null);
        try {
            // Update title
            const titleRes = await fetch(`/api/generation/${item.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title })
            });
            if (!titleRes.ok) {
                const errText = await titleRes.text();
                throw new Error(`Failed to update title: ${errText}`);
            }

            // Handle cover: remove, replace, or keep
            if (removeCover && !coverFile) {
                // User wants to remove cover
                const delRes = await fetch(`/api/generation/${item.id}/cover`, {
                    method: 'DELETE'
                });
                if (!delRes.ok && delRes.status !== 404) {
                    const errText = await delRes.text();
                    throw new Error(`Failed to remove cover: ${errText}`);
                }
            } else if (coverFile) {
                // Upload new cover
                const formData = new FormData();
                formData.append('file', coverFile);
                const uploadRes = await fetch(`/api/generation/${item.id}/cover`, {
                    method: 'POST',
                    body: formData
                });
                if (!uploadRes.ok) {
                    const errText = await uploadRes.text();
                    throw new Error(`Failed to upload cover: ${errText}`);
                }
            }

            onSave && onSave();
            onClose();
        } catch (e) {
            console.error('Save failed:', e);
            setError(e.message || 'Save failed');
        } finally {
            setSaving(false);
        }
    };

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.8)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
        }} onClick={onClose}>
            <div style={{
                backgroundColor: '#282828',
                borderRadius: '16px',
                padding: '24px',
                width: '400px',
                maxWidth: '90vw',
                border: '1px solid #3a3a3a',
            }} onClick={e => e.stopPropagation()}>
                <div style={{ fontSize: '18px', fontWeight: '600', color: '#e0e0e0', marginBottom: '20px' }}>
                    Edit Song
                </div>

                {/* Title Input */}
                <div style={{ marginBottom: '20px' }}>
                    <label style={{ display: 'block', fontSize: '12px', color: '#888', marginBottom: '8px' }}>Title</label>
                    <input
                        type="text"
                        value={title}
                        onChange={e => setTitle(e.target.value)}
                        style={{
                            width: '100%',
                            backgroundColor: '#1e1e1e',
                            border: '1px solid #3a3a3a',
                            borderRadius: '8px',
                            padding: '12px',
                            color: '#e0e0e0',
                            fontSize: '14px',
                            outline: 'none',
                            boxSizing: 'border-box',
                        }}
                        placeholder="Enter song title..."
                    />
                </div>

                {/* Album Cover Upload */}
                <div style={{ marginBottom: '24px' }}>
                    <label style={{ display: 'block', fontSize: '12px', color: '#888', marginBottom: '8px' }}>Album Cover</label>
                    <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                        {/* Preview */}
                        <div style={{
                            width: '100px',
                            height: '100px',
                            borderRadius: '8px',
                            backgroundColor: '#1e1e1e',
                            border: '1px solid #3a3a3a',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            overflow: 'hidden',
                            flexShrink: 0,
                            position: 'relative',
                        }}>
                            {coverPreview ? (
                                <img src={coverPreview} style={{ width: '100%', height: '100%', objectFit: 'cover' }} alt="New cover" />
                            ) : (!removeCover && hasCover && existingCoverUrl) ? (
                                <img src={existingCoverUrl} style={{ width: '100%', height: '100%', objectFit: 'cover' }} alt="Album cover" />
                            ) : (
                                <MusicNoteIcon size={32} color="#444" />
                            )}
                            {/* Trash icon - show when there's an image to remove */}
                            {(coverPreview || (!removeCover && hasCover && existingCoverUrl)) && (
                                <button
                                    onClick={handleRemoveCover}
                                    title="Remove cover"
                                    style={{
                                        position: 'absolute',
                                        top: '4px',
                                        right: '4px',
                                        width: '22px',
                                        height: '22px',
                                        borderRadius: '50%',
                                        backgroundColor: 'rgba(0,0,0,0.7)',
                                        border: 'none',
                                        cursor: 'pointer',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        padding: 0,
                                        transition: 'background-color 0.15s',
                                    }}
                                    onMouseEnter={e => e.target.style.backgroundColor = 'rgba(239,68,68,0.9)'}
                                    onMouseLeave={e => e.target.style.backgroundColor = 'rgba(0,0,0,0.7)'}
                                >
                                    <TrashIcon size={12} color="#fff" />
                                </button>
                            )}
                        </div>
                        {/* Upload Button */}
                        <div style={{ flex: 1 }}>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/*"
                                onChange={handleFileSelect}
                                style={{ display: 'none' }}
                            />
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                style={{
                                    width: '100%',
                                    padding: '10px 16px',
                                    backgroundColor: '#1e1e1e',
                                    border: '1px dashed #3a3a3a',
                                    borderRadius: '8px',
                                    color: '#888',
                                    cursor: 'pointer',
                                    fontSize: '13px',
                                    marginBottom: '8px',
                                }}
                            >
                                Choose Image...
                            </button>
                            <div style={{ fontSize: '11px', color: '#555' }}>
                                JPG, PNG, GIF or WebP. Square recommended.
                            </div>
                        </div>
                    </div>
                </div>

                {/* Error Message */}
                {error && (
                    <div style={{
                        backgroundColor: 'rgba(239, 68, 68, 0.15)',
                        border: '1px solid rgba(239, 68, 68, 0.3)',
                        borderRadius: '8px',
                        padding: '10px 12px',
                        marginBottom: '16px',
                        color: '#f87171',
                        fontSize: '13px',
                    }}>
                        {error}
                    </div>
                )}

                {/* Buttons */}
                <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
                    <button
                        onClick={onClose}
                        style={{
                            padding: '10px 20px',
                            backgroundColor: '#1e1e1e',
                            border: '1px solid #3a3a3a',
                            borderRadius: '8px',
                            color: '#888',
                            cursor: 'pointer',
                            fontSize: '14px',
                        }}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={saving}
                        style={{
                            padding: '10px 20px',
                            backgroundColor: '#10B981',
                            border: 'none',
                            borderRadius: '8px',
                            color: '#fff',
                            cursor: saving ? 'not-allowed' : 'pointer',
                            fontSize: '14px',
                            fontWeight: '500',
                            opacity: saving ? 0.7 : 1,
                        }}
                    >
                        {saving ? 'Saving...' : 'Save'}
                    </button>
                </div>
            </div>
        </div>
    );
};

// Library item component
var LibraryItem = ({ item, isQueued, isGenerating, queuePosition, onRemoveFromQueue, onStop, onDelete, onPlay, onUpdate, isCurrentlyPlaying, isAudioPlaying, playingTrackIdx, status, elapsedTime, estimatedTime }) => {
    const [expanded, setExpanded] = useState(false);
    const [editing, setEditing] = useState(false);
    const [exportingMp4, setExportingMp4] = useState(false);
    const [exportError, setExportError] = useState(null);
    const [selectedTrack, setSelectedTrack] = useState(0);
    const meta = (isQueued || isGenerating) ? item : (item.metadata || {});

    // Track info for separate mode
    const trackLabels = item.track_labels || [];
    const hasMultipleTracks = (item.output_files?.length || 0) > 1 && trackLabels.length > 1;
    const hasReference = !!(meta.reference_audio || meta.reference_audio_id);

    // Use a stable cache key based on cover filename (changes when cover is updated)
    const coverCacheKey = meta.cover || '';

    // Initialize coverUrl from metadata if available (prevents flash on re-render)
    const [coverUrl, setCoverUrl] = useState(() => {
        if (!isQueued && !isGenerating && item.id && meta.cover) {
            return `/api/generation/${item.id}/cover?v=${encodeURIComponent(meta.cover)}`;
        }
        return null;
    });
    // Track the last cover key we processed to detect changes
    const lastCoverKeyRef = useRef(coverCacheKey);

    // Check for cover image - only update when cover actually changes
    // Trust metadata as source of truth - don't probe server if no cover in metadata
    useEffect(() => {
        if (!isQueued && !isGenerating && item.id) {
            const coverChanged = lastCoverKeyRef.current !== coverCacheKey;

            if (meta.cover) {
                // Only set new URL if cover changed or we don't have one yet
                if (coverChanged || !coverUrl) {
                    setCoverUrl(`/api/generation/${item.id}/cover?v=${encodeURIComponent(meta.cover)}`);
                    lastCoverKeyRef.current = coverCacheKey;
                }
            } else if (coverChanged) {
                // Cover was removed - clear the URL
                setCoverUrl(null);
                lastCoverKeyRef.current = coverCacheKey;
            }
        }
    }, [item.id, isQueued, isGenerating, coverCacheKey]);

    const formatDate = (dateStr) => {
        if (!dateStr) return 'Unknown';
        try {
            return new Date(dateStr).toLocaleString();
        } catch { return dateStr; }
    };

    const formatDuration = (start, end) => {
        if (!start || !end) return '';
        try {
            const ms = new Date(end) - new Date(start);
            const mins = Math.floor(ms / 60000);
            const secs = Math.floor((ms % 60000) / 1000);
            return `${mins}m ${secs}s`;
        } catch { return ''; }
    };

    const canPlay = !isQueued && !isGenerating && item.status === 'completed' && (item.output_files?.length > 0 || item.output_file);
    const albumClass = isCurrentlyPlaying ? 'playing' : isGenerating ? 'generating' : isQueued ? 'queued' : item.status === 'failed' ? 'failed' : 'default';

    // Export as MP4 video with progress feedback
    const exportMp4 = async () => {
        if (exportingMp4) return;
        setExportingMp4(true);
        setExportError(null);

        try {
            const response = await fetch(`/api/generation/${item.id}/video`);
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || 'Export failed');
            }

            // Get the blob and trigger download
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${item.title || meta.title || 'song'}.mp4`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error('MP4 export error:', err);
            setExportError(err.message);
        } finally {
            setExportingMp4(false);
        }
    };

    return (
        <div className="card-base" style={{
            padding: '12px',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
            border: isCurrentlyPlaying ? '1px solid #10B981' : '1px solid transparent',
        }}>
            {/* Main row */}
            <div className="flex gap-4">
            {/* Album Cover with Play Button */}
            <div
                onClick={() => canPlay && onPlay && onPlay(item)}
                className={`album-cover ${albumClass}`}
                style={{
                    cursor: canPlay ? 'pointer' : 'default',
                    backgroundImage: coverUrl ? `url(${coverUrl})` : 'none',
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                }}
            >
                {isGenerating ? (
                    <SpinnerIcon />
                ) : canPlay ? (
                    coverUrl ? (
                        <div className="album-cover-overlay">
                            {isCurrentlyPlaying && isAudioPlaying ? (
                                <PauseLargeIcon />
                            ) : (
                                <PlayLargeIcon style={{ marginLeft: '3px' }} />
                            )}
                        </div>
                    ) : (
                        isCurrentlyPlaying && isAudioPlaying ? (
                            <PauseLargeIcon />
                        ) : (
                            <PlayLargeIcon style={{ marginLeft: '3px' }} />
                        )
                    )
                ) : !coverUrl ? (
                    <MusicNoteIcon />
                ) : null}
            </div>

            {/* Song Info */}
            <div style={{ flex: 1, minWidth: 0 }} className="flex flex-col justify-center">
                <div className="flex items-center gap-3" style={{ marginBottom: '4px' }}>
                    <div className="text-lg font-medium text-primary truncate">{item.title || 'Untitled'}</div>
                    <span className={`tag ${isGenerating ? 'tag-warning' : isQueued ? 'tag-queue' : item.status === 'completed' ? 'tag-primary' : 'tag-error'}`} style={{ flexShrink: 0 }}>
                        {isGenerating ? 'generating' : (isQueued ? `#${queuePosition}` : item.status)}
                    </span>
                </div>
                <div className="text-sm text-secondary" style={{ marginBottom: '6px' }}>
                    {isGenerating ? (status || 'Generating...') : (isQueued ? `Queued` : ((item.duration || meta.duration) ? formatTime(item.duration || meta.duration) : '--:--'))}
                </div>
                {/* Style tags */}
                <div className="flex gap-1" style={{ flexWrap: 'wrap' }}>
                    {meta.gender && <span className="tag tag-info">{meta.gender}</span>}
                    {meta.genre && <span className="tag tag-purple">{meta.genre}</span>}
                    {meta.emotion && <span className="tag tag-warning">{meta.emotion}</span>}
                    {meta.bpm && <span className="tag tag-primary">{meta.bpm} BPM</span>}
                    {hasMultipleTracks && <span className="tag" style={{ backgroundColor: 'rgba(16, 185, 129, 0.2)', color: '#10B981' }}>Stems</span>}
                    {hasReference && <span className="tag" style={{ backgroundColor: 'rgba(139, 92, 246, 0.2)', color: '#8B5CF6' }}>Style Clone</span>}
                </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-2 items-center" style={{ flexShrink: 0 }}>
                {canPlay && (
                    <>
                        {['FLAC', 'MP3'].map((fmt) => (
                            <button key={fmt} className="btn-icon btn-success" onClick={() => {
                                const a = document.createElement('a');
                                a.href = `/api/audio/${item.id}/0?format=${fmt.toLowerCase()}`;
                                a.download = `${item.title || 'song'}.${fmt.toLowerCase()}`;
                                a.click();
                            }}>{fmt}</button>
                        ))}
                        <button
                            className="btn-icon btn-success"
                            onClick={exportMp4}
                            disabled={exportingMp4}
                            style={{ opacity: exportingMp4 ? 0.7 : 1, minWidth: '50px' }}
                            title={exportError ? `Error: ${exportError}` : 'Export as MP4 video with waveform visualization'}
                        >
                            {exportingMp4 ? (
                                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                    <span style={{
                                        width: '10px',
                                        height: '10px',
                                        border: '2px solid #10B981',
                                        borderTopColor: 'transparent',
                                        borderRadius: '50%',
                                        animation: 'spin 1s linear infinite',
                                        display: 'inline-block'
                                    }} />
                                </span>
                            ) : 'MP4'}
                        </button>
                    </>
                )}
                {/* Stop button - only for pending items (not actively processing - those can't be stopped) */}
                {onStop && !isQueued && item.status === 'pending' && (
                    <button className="btn-icon btn-danger" onClick={onStop}>Stop</button>
                )}
                {isQueued && <button className="btn-icon btn-danger" onClick={onRemoveFromQueue}>Remove</button>}
                {!isQueued && !isGenerating && item.status === 'completed' && (
                    <button className="btn-icon" onClick={() => setEditing(true)} title="Edit"><EditIcon /></button>
                )}
                <button className="btn-icon" onClick={() => setExpanded(!expanded)}>{expanded ? 'Hide' : 'Details'}</button>
                {!isQueued && !isGenerating && (item.status === 'completed' || item.status === 'failed' || item.status === 'stopped') && (
                    <button className="btn-icon" onClick={onDelete} title="Delete"><TrashIcon /></button>
                )}
            </div>
            </div>

            {/* Track Selector for Separate Mode */}
            {hasMultipleTracks && canPlay && (
                <div style={{
                    display: 'flex',
                    gap: '8px',
                    alignItems: 'center',
                    padding: '8px 12px',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderRadius: '8px',
                    border: '1px solid rgba(16, 185, 129, 0.2)',
                }}>
                    <span style={{ color: '#10B981', fontSize: '12px', fontWeight: '500' }}>Tracks:</span>
                    {trackLabels.map((label, idx) => {
                        const isThisTrackPlaying = isCurrentlyPlaying && playingTrackIdx === idx;
                        return (
                            <button
                                key={idx}
                                onClick={() => onPlay && onPlay(item, idx)}
                                style={{
                                    padding: '6px 12px',
                                    borderRadius: '6px',
                                    border: isThisTrackPlaying ? '1px solid #10B981' : '1px solid #333',
                                    backgroundColor: isThisTrackPlaying ? 'rgba(16, 185, 129, 0.2)' : '#1a1a1a',
                                    color: isThisTrackPlaying ? '#10B981' : '#888',
                                    fontSize: '12px',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '6px',
                                    transition: 'all 0.15s ease',
                                }}
                            >
                                {isThisTrackPlaying && isAudioPlaying && (
                                    <span style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
                                        <span style={{ width: '3px', height: '10px', backgroundColor: '#10B981', borderRadius: '1px', animation: 'pulse 0.5s ease-in-out infinite alternate' }}></span>
                                        <span style={{ width: '3px', height: '14px', backgroundColor: '#10B981', borderRadius: '1px', animation: 'pulse 0.5s ease-in-out infinite alternate', animationDelay: '0.1s' }}></span>
                                        <span style={{ width: '3px', height: '8px', backgroundColor: '#10B981', borderRadius: '1px', animation: 'pulse 0.5s ease-in-out infinite alternate', animationDelay: '0.2s' }}></span>
                                    </span>
                                )}
                                {label}
                            </button>
                        );
                    })}
                    <div style={{ flex: 1 }}></div>
                    <span style={{ color: '#666', fontSize: '11px' }}>Download:</span>
                    {trackLabels.map((label, idx) => (
                        <button
                            key={`dl-${idx}`}
                            className="btn-icon btn-success"
                            style={{ padding: '4px 8px', fontSize: '10px' }}
                            onClick={() => {
                                const a = document.createElement('a');
                                a.href = `/api/audio/${item.id}/${idx}?format=flac`;
                                a.download = `${item.title || 'song'}_${label.toLowerCase().replace(' ', '_')}.flac`;
                                a.click();
                            }}
                            title={`Download ${label}`}
                        >
                            {label === 'Full Song' ? 'Full' : label === 'Vocals' ? 'Voc' : 'Inst'}
                        </button>
                    ))}
                </div>
            )}

            {/* Edit Modal */}
            {editing && (
                <EditModal
                    item={item}
                    onClose={() => setEditing(false)}
                    onSave={() => {
                        // Trigger library reload to get updated metadata
                        onUpdate && onUpdate();
                    }}
                />
            )}

            {/* MP4 Export Error Message */}
            {exportError && (
                <div style={{
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    borderRadius: '8px',
                    padding: '10px 14px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ color: '#EF4444', fontWeight: '500', fontSize: '13px' }}>MP4 Export Failed:</span>
                        <span style={{ color: '#f87171', fontSize: '12px' }}>{exportError}</span>
                    </div>
                    <button
                        onClick={() => setExportError(null)}
                        style={{ background: 'none', border: 'none', color: '#EF4444', cursor: 'pointer', padding: '4px' }}
                    >
                        <CloseIcon size={14} />
                    </button>
                </div>
            )}

            {/* Expanded details */}
            {expanded && (
                <div style={{
                    backgroundColor: '#1a1a1a',
                    borderRadius: '8px',
                    padding: '12px',
                    fontSize: '12px',
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)',
                    gap: '8px 16px',
                }}>
                    <div><span style={{ color: '#666' }}>Model:</span> <span style={{ color: '#aaa' }}>{meta.model || 'unknown'}</span></div>
                    <div><span style={{ color: '#666' }}>Output Mode:</span> <span style={{ color: '#aaa' }}>{meta.output_mode || 'mixed'}</span></div>
                    <div><span style={{ color: '#666' }}>Voice:</span> <span style={{ color: '#aaa' }}>{meta.gender || '-'}</span></div>
                    <div><span style={{ color: '#666' }}>BPM:</span> <span style={{ color: '#aaa' }}>{meta.bpm || '-'}</span></div>
                    <div><span style={{ color: '#666' }}>Genre:</span> <span style={{ color: '#aaa' }}>{meta.genre || '-'}</span></div>
                    <div><span style={{ color: '#666' }}>Mood:</span> <span style={{ color: '#aaa' }}>{meta.emotion || '-'}</span></div>
                    <div><span style={{ color: '#666' }}>Timbre:</span> <span style={{ color: '#aaa' }}>{meta.timbre || '-'}</span></div>
                    <div><span style={{ color: '#666' }}>Instruments:</span> <span style={{ color: '#aaa' }}>{meta.instruments || '-'}</span></div>
                    {meta.custom_style && <div style={{ gridColumn: '1 / -1' }}><span style={{ color: '#666' }}>Custom Style:</span> <span style={{ color: '#aaa' }}>{meta.custom_style}</span></div>}
                    {!isQueued && !isGenerating && <div style={{ gridColumn: '1 / -1' }}><span style={{ color: '#666' }}>Created:</span> <span style={{ color: '#aaa' }}>{formatDate(meta.created_at)}</span></div>}
                    {!isQueued && !isGenerating && meta.completed_at && (
                        <div style={{ gridColumn: '1 / -1' }}><span style={{ color: '#666' }}>Completed:</span> <span style={{ color: '#aaa' }}>{formatDate(meta.completed_at)} ({formatDuration(meta.created_at, meta.completed_at)})</span></div>
                    )}
                    {meta.description && (
                        <div style={{ gridColumn: '1 / -1', marginTop: '8px' }}>
                            <div style={{ color: '#666', marginBottom: '4px' }}>Description sent to model:</div>
                            <div style={{ color: '#10B981', fontFamily: 'monospace', fontSize: '11px', padding: '8px', backgroundColor: '#0a0a0a', borderRadius: '4px' }}>{meta.description}</div>
                        </div>
                    )}
                    {meta.sections && meta.sections.length > 0 && (
                        <div style={{ gridColumn: '1 / -1', marginTop: '8px' }}>
                            <div style={{ color: '#666', marginBottom: '4px' }}>Sections ({meta.sections.length}):</div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                {meta.sections.map((s, i) => (
                                    <div key={i} style={{ padding: '8px 10px', backgroundColor: '#0a0a0a', borderRadius: '4px', borderLeft: `3px solid ${SECTION_TYPES[s.type]?.color || '#666'}` }}>
                                        <div style={{ color: SECTION_TYPES[s.type]?.color || '#888', fontWeight: '500', marginBottom: s.lyrics ? '6px' : '0' }}>[{s.type}]</div>
                                        {s.lyrics && <div style={{ color: '#999', whiteSpace: 'pre-wrap', lineHeight: '1.5', fontSize: '11px' }}>{s.lyrics}</div>}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                    {(meta.reference_audio || meta.reference_audio_id) && (
                        <div style={{ gridColumn: '1 / -1' }}>
                            <div style={{ color: '#666', marginBottom: '8px' }}>Reference Audio (style cloning):</div>
                            <DarkAudioPlayer src={`/api/reference/${meta.reference_audio_id || meta.reference_audio?.replace(/\\/g, '/').split('/').pop()?.split('_')[0]}`} />
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

// Songs Panel Item - with hover-based play button overlay
var SongsPanelItem = ({ item, audioPlayer, onDelete }) => {
    const [isHovered, setIsHovered] = useState(false);
    const meta = item.metadata || {};
    const isPlaying = audioPlayer.playingId === item.id;
    const isAudioPlaying = audioPlayer.isPlaying;
    const canPlay = item.status === 'completed' && (item.output_file || item.output_files?.length > 0);
    const coverUrl = meta.cover ? `/api/generation/${item.id}/cover?v=${encodeURIComponent(meta.cover)}` : null;
    const hasReference = !!(meta.reference_audio || meta.reference_audio_id);
    const hasSeparateTracks = (item.track_labels?.length || 0) > 1 || item.output_mode === 'separate';

    const showOverlay = canPlay && coverUrl && (isHovered || (isPlaying && isAudioPlaying));

    return (
        <div
            className={`activity-item ${isPlaying ? 'active' : ''} ${item.status === 'failed' ? 'error' : ''}`}
            style={{ cursor: canPlay ? 'pointer' : 'default', display: 'flex', gap: '10px', alignItems: 'center', position: 'relative' }}
            onClick={() => canPlay && audioPlayer.play(item)}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            {['completed', 'failed', 'stopped'].includes(item.status) && (
                <button onClick={(e) => { e.stopPropagation(); onDelete(); }} style={{ position: 'absolute', top: '4px', right: '4px', background: 'none', border: 'none', color: '#555', cursor: 'pointer' }}>
                    <TrashIcon size={10} />
                </button>
            )}
            <div style={{
                width: '44px', height: '44px', borderRadius: '6px',
                backgroundColor: isPlaying ? '#10B981' : item.status === 'failed' ? '#EF4444' : '#2a2a2a',
                backgroundImage: coverUrl ? `url(${coverUrl})` : 'none',
                backgroundSize: 'cover', backgroundPosition: 'center',
                display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative'
            }}>
                {canPlay && !coverUrl && (
                    isPlaying && isAudioPlaying ? <PauseLargeIcon size={16} color="#fff" /> : <PlayLargeIcon size={16} color={isPlaying ? '#fff' : '#888'} />
                )}
                {showOverlay && (
                    <div style={{
                        position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        backgroundColor: 'rgba(0, 0, 0, 0.5)', borderRadius: '6px'
                    }}>
                        {isPlaying && isAudioPlaying ? <PauseLargeIcon size={16} color="#fff" /> : <PlayLargeIcon size={16} color="#fff" style={{ marginLeft: '2px' }} />}
                    </div>
                )}
                {item.status === 'failed' && <CloseIcon color="#fff" />}
                {/* Separate tracks indicator */}
                {hasSeparateTracks && (
                    <div style={{
                        position: 'absolute', bottom: '-2px', left: '-2px',
                        width: '14px', height: '14px', borderRadius: '50%',
                        backgroundColor: '#10B981', border: '2px solid #1e1e1e',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '8px', color: '#fff', fontWeight: 'bold'
                    }} title="Has separate vocal & instrumental tracks">3</div>
                )}
                {/* Reference audio indicator */}
                {hasReference && (
                    <div style={{
                        position: 'absolute', bottom: '-2px', right: '-2px',
                        width: '14px', height: '14px', borderRadius: '50%',
                        backgroundColor: '#8B5CF6', border: '2px solid #1e1e1e',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '8px', color: '#fff'
                    }} title="Style cloned from reference">R</div>
                )}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
                <div className="text-sm font-medium text-primary truncate">{meta.title || 'Untitled'}</div>
                <div className="text-xs text-secondary truncate">{[meta.genre, meta.emotion].filter(Boolean).join(' • ') || 'No tags'}</div>
                {(item.duration || meta.duration) && <div className="text-xs" style={{ color: '#555', marginTop: '2px' }}>{formatTime(item.duration || meta.duration)}</div>}
            </div>
        </div>
    );
};
