import io
import multiprocessing
import multiprocessing.queues
from multiprocessing.reduction import ForkingPickler
import pickle


class ConnectionWrapper(object):
    """Proxy class for _multiprocessing.Connection which uses ForkingPickler to
    serialize objects"""

    def __init__(self, conn):
        self.conn = conn

    def send(self, obj):
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        self.send_bytes(buf.getvalue())

    def recv(self):
        buf = self.recv_bytes()
        return pickle.loads(buf)

    def __getattr__(self, name):
        return getattr(self.conn, name)


class Queue(multiprocessing.queues.Queue):

    def __init__(self, *args, **kwargs):
        super(Queue, self).__init__(*args, **kwargs)
        self._reader = ConnectionWrapper(self._reader)
        self._writer = ConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv


class SimpleQueue(multiprocessing.queues.SimpleQueue):

    def _make_methods(self):
        if not isinstance(self._reader, ConnectionWrapper):
            self._reader = ConnectionWrapper(self._reader)
            self._writer = ConnectionWrapper(self._writer)
        super(SimpleQueue, self)._make_methods()
