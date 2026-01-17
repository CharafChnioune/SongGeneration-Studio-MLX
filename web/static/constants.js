// SongGeneration Studio - Constants & Configuration

// React hooks extraction
var { useState, useEffect, useRef, useCallback, useMemo } = React;

// Section type configurations
var SECTION_TYPES = {
    'intro': { name: 'Intro', color: '#8B5CF6', hasLyrics: false, hasDuration: true },
    'verse': { name: 'Verse', color: '#3B82F6', hasLyrics: true, hasDuration: false },
    'chorus': { name: 'Chorus', color: '#F59E0B', hasLyrics: true, hasDuration: false },
    'bridge': { name: 'Bridge', color: '#EC4899', hasLyrics: true, hasDuration: false },
    'inst': { name: 'Instrumental', color: '#10B981', hasLyrics: false, hasDuration: true },
    'outro': { name: 'Outro', color: '#EAB308', hasLyrics: false, hasDuration: true },
};

// Model generation time defaults (in seconds) - used as starting point before learning
var MODEL_BASE_TIMES = {
    'songgeneration_base': 180,       // 3:00
    'songgeneration_base_new': 180,   // 3:00
    'songgeneration_base_full': 240,  // 4:00
    'songgeneration_large': 360,      // 6:00
};

// Maximum additional time based on lyrics/sections (in seconds)
var MODEL_MAX_ADDITIONAL = {
    'songgeneration_base': 180,       // +3:00 max = 6:00 total
    'songgeneration_base_new': 180,   // +3:00 max = 6:00 total
    'songgeneration_base_full': 240,  // +4:00 max = 8:00 total
    'songgeneration_large': 360,      // +6:00 max = 12:00 total
};

// Default song sections
var DEFAULT_SECTIONS = [
    { id: '1', type: 'intro-short', lyrics: '' },
    { id: '2', type: 'verse', lyrics: '' },
    { id: '3', type: 'chorus', lyrics: '' },
    { id: '4', type: 'verse', lyrics: '' },
    { id: '5', type: 'outro-short', lyrics: '' },
];

// Duration widths for resizable sections
var DURATION_WIDTHS = { short: 68, medium: 92 };

// Suggestion lists
var GENRE_SUGGESTIONS = [
    'pop',
    'electronic',
    'hip hop',
    'rock',
    'jazz',
    'blues',
    'classical',
    'rap',
    'country',
    'classic rock',
    'hard rock',
    'folk',
    'soul',
    'dance, electronic',
    'rockabilly',
    'dance, dancepop, house, pop',
    'reggae',
    'experimental',
    'dance, pop',
    'dance, deephouse, electronic',
    'k-pop',
    'experimental pop',
    'pop punk',
    'rock and roll',
    'R&B',
    'varies',
    'pop rock',
];
var MOOD_SUGGESTIONS = [
    'sad',
    'emotional',
    'angry',
    'happy',
    'uplifting',
    'intense',
    'romantic',
    'melancholic',
];
var TIMBRE_SUGGESTIONS = [
    'dark',
    'bright',
    'warm',
    'rock',
    'varies',
    'soft',
    'vocal',
];
var INSTRUMENT_SUGGESTIONS = [
    'synthesizer and piano',
    'piano and drums',
    'piano and synthesizer',
    'synthesizer and drums',
    'piano and strings',
    'guitar and drums',
    'guitar and piano',
    'piano and double bass',
    'piano and guitar',
    'acoustic guitar and piano',
    'acoustic guitar and synthesizer',
    'synthesizer and guitar',
    'piano and saxophone',
    'saxophone and piano',
    'piano and violin',
    'electric guitar and drums',
    'acoustic guitar and drums',
    'synthesizer',
    'guitar and fiddle',
    'guitar and harmonica',
    'synthesizer and acoustic guitar',
    'beats',
    'piano',
    'acoustic guitar and fiddle',
    'brass and piano',
    'bass and drums',
    'violin',
    'acoustic guitar and harmonica',
    'piano and cello',
    'saxophone and trumpet',
    'guitar and banjo',
    'guitar and synthesizer',
    'saxophone',
    'violin and piano',
    'synthesizer and bass',
    'synthesizer and electric guitar',
    'electric guitar and piano',
    'beats and piano',
    'guitar',
];

// ============ Utility Functions ============

// Format seconds to MM:SS
var formatTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
};

// Format seconds to M:SS or H:MM:SS
var formatEta = (seconds) => {
    if (!seconds || isNaN(seconds) || seconds < 0) return '--:--';
    const total = Math.floor(seconds);
    const hours = Math.floor(total / 3600);
    const mins = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    if (hours > 0) {
        return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
};

// Parse section type into base and duration
var fromApiType = (type) => {
    const parts = type.split('-');
    if (parts.length === 2 && ['short', 'medium'].includes(parts[1])) {
        return { base: parts[0], duration: parts[1] };
    }
    if (parts.length === 2 && parts[1] === 'long') {
        return { base: parts[0], duration: 'medium' };
    }
    return { base: type, duration: null };
};

// Scrollbar style helper
var getScrollStyle = (isHovered) => ({
    scrollbarWidth: 'thin',
    scrollbarColor: isHovered ? '#3a3a3a transparent' : 'transparent transparent',
});

// Button style generator
var btnStyle = (isActive, activeColor = '#10B981') => ({
    flex: 1,
    padding: '10px 16px',
    borderRadius: '10px',
    border: 'none',
    fontSize: '13px',
    fontWeight: '500',
    cursor: 'pointer',
    transition: 'all 0.15s',
    backgroundColor: isActive ? activeColor : '#1e1e1e',
    color: isActive ? '#fff' : '#777',
});
