from collections import deque
import threading

class BaseStreamer:
    def __init__(self):
        self.queue = deque()
        self.closed = False
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)

    def put(self, data):
        with self.lock:
            if not self.closed:
                self.queue.append(data)
                self.not_empty.notify()

    def pop(self):
        with self.lock:
            while True:
                if self.queue:
                    return self.queue.popleft()
                if self.closed:
                    return None
                self.not_empty.wait()

    def close(self):
        with self.lock:
            self.closed = True
            self.not_empty.notify_all()