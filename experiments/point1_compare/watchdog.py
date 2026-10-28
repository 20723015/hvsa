"""Hard round deadline: preserve evidence, then terminate the isolated driver.

The caller supplies the persistence callback. A timeout never fabricates a
completed round or silently restarts a protocol with different keys/state.
"""
from __future__ import annotations
import os
import threading
import time


class RoundDeadline:
    def __init__(self, seconds, on_timeout):
        if seconds <= 0:
            raise ValueError("Positive deadline required")
        self.seconds, self.on_timeout = seconds, on_timeout
        self._closed = False
        self._lock = threading.Lock()
        self._timer = threading.Timer(seconds, self._expire)
        self._timer.daemon = True

    def _expire(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.on_timeout()
            finally:
                os._exit(124)

    def __enter__(self):
        self._started = time.perf_counter()
        self._timer.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._timer.cancel()
        # Also classify a late return if a native call held the interpreter
        # lock until after the deadline and delayed the watchdog thread.
        if exc_type is None and time.perf_counter()-self._started >= self.seconds:
            self._expire()
        with self._lock:
            self._closed = True
        return False
