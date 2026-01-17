// SongGeneration Studio - API Services

// ============ Generation API ============
var startGeneration = async (payload) => {
    const r = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
};

var fetchGeneration = async (id) => {
    const r = await fetch(`/api/generation/${id}`);
    if (!r.ok) {
        if (r.status === 404) return null;
        throw new Error(`Failed to fetch generation: ${r.status}`);
    }
    return r.json();
};

var stopGeneration = async (id) => {
    await fetch(`/api/stop/${id}`, { method: 'POST' });
};

var deleteGeneration = async (id) => {
    const r = await fetch(`/api/generation/${id}`, { method: 'DELETE' });
    return r.ok;
};

// ============ Library API ============
var fetchLibrary = async () => {
    const r = await fetch('/api/generations');
    if (!r.ok) throw new Error(`Failed to load library: ${r.status}`);
    const data = await r.json();
    return data.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
};

// ============ Queue API ============
var fetchQueue = async () => {
    const r = await fetch('/api/queue');
    if (!r.ok) return [];
    return r.json();
};

var addToQueue = async (payload) => {
    const r = await fetch('/api/queue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    return r.ok;
};

var removeFromQueue = async (itemId) => {
    await fetch(`/api/queue/${itemId}`, { method: 'DELETE' });
};

// ============ Other API ============
var fetchGpuInfo = async () => {
    const r = await fetch('/api/gpu');
    if (!r.ok) return null;
    return r.json();
};

var fetchTimingStats = async () => {
    const r = await fetch('/api/timing-stats');
    if (!r.ok) return null;
    return r.json();
};
