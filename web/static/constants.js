// SongGeneration Studio - Constants & Configuration

// React hooks extraction
var { useState, useEffect, useRef, useCallback, useMemo } = React;

// Section type configurations
var SECTION_TYPES = {
    'intro': { name: 'Intro', color: '#8B5CF6', hasLyrics: true, hasDuration: true },
    'verse': { name: 'Verse', color: '#3B82F6', hasLyrics: true, hasDuration: false },
    'chorus': { name: 'Chorus', color: '#F59E0B', hasLyrics: true, hasDuration: false },
    'bridge': { name: 'Bridge', color: '#EC4899', hasLyrics: true, hasDuration: false },
    'inst': { name: 'Inst', color: '#10B981', hasLyrics: false, hasDuration: true },
    'outro': { name: 'Outro', color: '#EAB308', hasLyrics: true, hasDuration: true },
    'prechorus': { name: 'Pre-Chorus', color: '#06B6D4', hasLyrics: true, hasDuration: false },
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

// Arrangement templates for auto-structuring songs
var ARRANGEMENT_TEMPLATES = [
    {
        id: 'auto',
        name: 'Auto (by genre)',
        description: 'Matches your selected genre.',
        sections: [],
    },
    {
        id: 'pop_hit',
        name: 'Pop Hit',
        description: 'Hook-forward radio structure.',
        sections: ['intro-short', 'verse', 'prechorus', 'chorus', 'verse', 'prechorus', 'chorus', 'bridge', 'chorus', 'outro-short'],
    },
    {
        id: 'hiphop_story',
        name: 'Hip-Hop Story',
        description: 'Verse-driven with a strong hook.',
        sections: ['intro-short', 'verse', 'chorus', 'verse', 'chorus', 'verse', 'outro-short'],
    },
    {
        id: 'trap_bounce',
        name: 'Trap Bounce',
        description: 'Shorter verses, punchy hooks.',
        sections: ['intro-short', 'verse', 'chorus', 'verse', 'chorus', 'outro-short'],
    },
    {
        id: 'rnb_smooth',
        name: 'R&B Smooth',
        description: 'Melodic verses with a bridge lift.',
        sections: ['intro-short', 'verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus', 'outro-short'],
    },
    {
        id: 'rock_anthem',
        name: 'Rock Anthem',
        description: 'Big chorus, extended outro.',
        sections: ['intro-short', 'verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus', 'outro-long'],
    },
    {
        id: 'edm_build',
        name: 'EDM Build',
        description: 'Build + drop + outro.',
        sections: ['intro-short', 'inst-medium', 'verse', 'prechorus', 'chorus', 'inst-short', 'chorus', 'outro-long'],
    },
];

var GENRE_TO_TEMPLATE = {
    'hiphop': 'hiphop_story',
    'hip-hop': 'hiphop_story',
    'rap': 'hiphop_story',
    'trap': 'trap_bounce',
    'pop': 'pop_hit',
    'rock': 'rock_anthem',
    'metal': 'rock_anthem',
    'r&b': 'rnb_smooth',
    'rnb': 'rnb_smooth',
    'edm': 'edm_build',
    'dance': 'edm_build',
    'electronic': 'edm_build',
};

// Duration widths for resizable sections
var DURATION_WIDTHS = { short: 52, medium: 66, long: 80 };

// Suggestion lists
var GENRE_SUGGESTIONS = [
    'Pop', 'Pop Rock', 'Indie Pop', 'Synth Pop', 'K-Pop', 'J-Pop', 'Latin Pop',
    'Rock', 'Alternative Rock', 'Indie Rock', 'Hard Rock', 'Punk', 'Pop Punk', 'Post-Punk',
    'Metal', 'Metalcore', 'Death Metal', 'Black Metal',
    'Hip-Hop', 'Rap', 'Trap', 'Drill', 'Boom Bap', 'G-Funk', 'Grime', 'Jersey Club', 'Phonk', 'Lo-Fi', 'Chillhop',
    'R&B', 'Neo Soul', 'Soul', 'Funk',
    'Electronic', 'EDM', 'House', 'Deep House', 'Tech House', 'Progressive House', 'Electro House',
    'Techno', 'Trance', 'Dubstep', 'Drum & Bass', 'Breakbeat', 'Garage', 'UK Garage', 'IDM', 'Glitch',
    'Ambient', 'Downtempo', 'Trip-Hop', 'Synthwave', 'Vaporwave', 'Future Bass',
    'Disco', 'Reggae', 'Dancehall', 'Reggaeton', 'Afrobeat', 'Amapiano', 'Salsa', 'Bossa Nova', 'Cumbia',
    'Country', 'Americana', 'Bluegrass', 'Folk', 'Singer-Songwriter', 'Acoustic',
    'Jazz', 'Jazz Fusion', 'Blues', 'Classical', 'Orchestral', 'Cinematic', 'Soundtrack', 'World',
];
var MOOD_SUGGESTIONS = [
    'Happy', 'Sad', 'Energetic', 'Calm', 'Romantic', 'Melancholic', 'Uplifting', 'Dark', 'Dreamy', 'Aggressive',
    'Peaceful', 'Nostalgic', 'Hopeful', 'Intense', 'Playful', 'Mysterious', 'Euphoric', 'Chill', 'Powerful', 'Tender',
    'Gritty', 'Moody', 'Angry', 'Confident', 'Anthemic', 'Cinematic', 'Ethereal', 'Luxurious', 'Raw', 'Spooky',
    'Lonely', 'Warm', 'Cold', 'Sexy', 'Rebellious', 'Mellow', 'Trippy', 'Hypnotic', 'Minimal', 'Introspective',
    'Suspenseful', 'Triumphant', 'Carefree', 'Somber', 'Haunting', 'Vulnerable', 'Optimistic', 'Dramatic',
];
var TIMBRE_SUGGESTIONS = [
    'Warm', 'Bright', 'Dark', 'Soft', 'Harsh', 'Smooth', 'Gritty', 'Airy', 'Rich', 'Thin', 'Full', 'Hollow',
    'Crisp', 'Mellow', 'Punchy', 'Breathy', 'Raspy', 'Nasal', 'Glassy', 'Metallic', 'Woody', 'Velvet',
    'Silky', 'Grainy', 'Crunchy', 'Thick', 'Dry', 'Wet', 'Sparkly', 'Muted', 'Open', 'Compressed', 'Dynamic', 'Resonant',
];
var INSTRUMENT_SUGGESTIONS = [
    'Piano', 'Electric Piano', 'Rhodes', 'Wurlitzer', 'Organ', 'Accordion', 'Synthesizer', 'Pad', 'Lead Synth', 'Arpeggiator', 'Keys',
    'Guitar', 'Electric Guitar', 'Acoustic Guitar', 'Bass', 'Electric Bass', 'Sub Bass', 'Synth Bass', '808',
    'Drums', 'Kick', 'Snare', 'Clap', 'Hi-Hats', 'Toms', 'Cymbals', 'Percussion', 'Shakers', 'Tambourine', 'Drum Machine',
    'Strings', 'Violin', 'Viola', 'Cello', 'Contrabass', 'Harp',
    'Brass', 'Trumpet', 'Trombone', 'Horn', 'Saxophone',
    'Woodwinds', 'Flute', 'Clarinet', 'Oboe', 'Bassoon',
    'Choir', 'Vocoder', 'Talkbox',
    'Bell', 'Glockenspiel', 'Marimba', 'Xylophone', 'Vibraphone', 'Steel Drums',
    'Sitar', 'Banjo', 'Mandolin', 'Ukulele', 'Harmonica', 'Bagpipes', 'Erhu', 'Shamisen',
    'Sampler', 'Turntables', 'Scratch', 'FX', 'Risers',
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
    if (parts.length === 2 && ['short', 'medium', 'long'].includes(parts[1])) {
        return { base: parts[0], duration: parts[1] };
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
