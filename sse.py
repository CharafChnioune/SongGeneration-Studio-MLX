"""
SongGeneration Studio - Event Broadcasting
Notification system for state changes (SSE disabled, used for internal notifications).
"""

import json
import threading
import queue as queue_module
from typing import List

from config import load_queue


# ============================================================================
# Broadcast System (SSE clients list kept for potential future use)
# ============================================================================

sse_clients: List[queue_module.Queue] = []
sse_lock = threading.Lock()


def broadcast_event(event_type: str, data: dict):
    """Broadcast an event to all connected SSE clients."""
    event_data = json.dumps({"type": event_type, **data})
    message = f"event: {event_type}\ndata: {event_data}\n\n"

    with sse_lock:
        dead_clients = []
        for client_queue in sse_clients:
            try:
                client_queue.put_nowait(message)
            except queue_module.Full:
                dead_clients.append(client_queue)
        for dead in dead_clients:
            sse_clients.remove(dead)


def notify_queue_update():
    """Notify all clients that queue changed."""
    broadcast_event("queue", {"queue": load_queue()})


def notify_generation_update(gen_id: str, gen_data: dict):
    """Notify all clients about generation status change."""
    broadcast_event("generation", {"id": gen_id, "generation": gen_data})


def notify_library_update(generations: dict):
    """Notify all clients that library changed."""
    summary = [{"id": g["id"], "status": g.get("status"), "progress": g.get("progress", 0)}
               for g in generations.values()]
    broadcast_event("library", {"generations": summary})


async def notify_models_update(get_all_models_func):
    """Notify all clients that model status changed."""
    all_models = await get_all_models_func()
    ready_models = [m for m in all_models if m["status"] == "ready"]
    broadcast_event("models", {
        "models": all_models,
        "ready_models": ready_models,
        "has_ready_model": len(ready_models) > 0
    })


def notify_models_update_sync():
    """Sync version for contexts where async isn't available."""
    from models import MODEL_REGISTRY, get_model_status_quick
    
    models = []
    for model_id, info in MODEL_REGISTRY.items():
        status = get_model_status_quick(model_id)
        models.append({
            "id": model_id,
            "name": info["name"],
            "status": status,
            "warmth": "not_loaded"
        })
    ready_models = [m for m in models if m["status"] == "ready"]
    broadcast_event("models", {
        "models": models,
        "ready_models": ready_models,
        "has_ready_model": len(ready_models) > 0
    })


