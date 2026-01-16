"""Streaming helpers for MLX modules."""

from contextlib import contextmanager
import typing as tp

import mlx.nn as nn

State = tp.Dict[str, tp.Any]


class StreamingModule(nn.Module):
    """Common API for streaming components.

    Each streaming component has a streaming state, which is just a dict[str, Any].
    If `_is_streaming` is True, the component should use and update `_streaming_state`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: State = {}
        self._is_streaming = False

    def _apply_named_streaming(self, fn: tp.Any) -> None:
        for name, module in self.named_modules():
            if isinstance(module, StreamingModule):
                fn(name, module)

    def _set_streaming(self, streaming: bool) -> None:
        def _set_streaming(name: str, module: "StreamingModule") -> None:
            module._is_streaming = streaming
        self._apply_named_streaming(_set_streaming)

    @contextmanager
    def streaming(self):
        """Context manager to enter streaming mode. Reset streaming state on exit."""
        self._set_streaming(True)
        try:
            yield
        finally:
            self._set_streaming(False)
            self.reset_streaming()

    def reset_streaming(self) -> None:
        """Reset the streaming state."""
        def _reset(name: str, module: "StreamingModule") -> None:
            module._streaming_state.clear()
        self._apply_named_streaming(_reset)

    def get_streaming_state(self) -> State:
        """Return the streaming state, including that of sub-modules."""
        state: State = {}

        def _add(name: str, module: "StreamingModule") -> None:
            if name:
                name += "."
            for key, value in module._streaming_state.items():
                state[name + key] = value

        self._apply_named_streaming(_add)
        return state

    def set_streaming_state(self, state: State) -> None:
        """Set the streaming state, including that of sub-modules."""
        state = dict(state)

        def _set(name: str, module: "StreamingModule") -> None:
            if name:
                name += "."
            module._streaming_state.clear()
            for key, value in list(state.items()):
                if key.startswith(name):
                    local_key = key[len(name):]
                    if "." not in local_key:
                        module._streaming_state[local_key] = value
                        del state[key]

        self._apply_named_streaming(_set)
        if state:
            raise ValueError(f"Unused streaming state keys: {list(state.keys())}")

    def flush(self, x: tp.Optional[tp.Any] = None):
        if x is None:
            return None
        return self(x)
