import queue
import sounddevice as sd
import numpy as np
import logging

logger = logging.getLogger("audio_capture")


class AudioCapture:
    """Captures audio from a microphone in real-time using sounddevice."""

    def __init__(self, sample_rate=16000, chunk_duration_sec=3):
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.chunk_samples = int(self.sample_rate * self.chunk_duration_sec)
        self.q = queue.Queue()
        self.stream = None
        self.is_recording = False
        self.buffer = np.array([], dtype=np.float32)

    def _callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio capture status: {status}")
        self.q.put(indata.copy())

    def start(self, device=None):
        if self.is_recording:
            return
        self.is_recording = True
        self.buffer = np.array([], dtype=np.float32)
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            device=device,
            callback=self._callback
        )
        self.stream.start()
        logger.info(f"Started audio stream: {self.sample_rate}Hz, channels=1")

    def stop(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logger.info("Stopped audio stream.")
        while not self.q.empty():
            self.q.get()

    def get_chunk(self):
        """Returns a chunk of audio of length chunk_duration_sec, or None if not enough data yet."""
        if not self.is_recording:
            return None

        try:
            while True:
                data = self.q.get_nowait()
                self.buffer = np.append(self.buffer, data)
        except queue.Empty:
            pass

        if len(self.buffer) >= self.chunk_samples:
            chunk = self.buffer[:self.chunk_samples]
            self.buffer = self.buffer[self.chunk_samples:]

            max_amp = np.max(np.abs(chunk))
            logger.info(f"Audio chunk prepared. Max amplitude: {max_amp:.5f}")

            if max_amp < 0.010:
                logger.info("Chunk dropped (detected as silence).")
                return None

            return chunk

        return None
