from dataclasses import dataclass
import subprocess
import time
import requests
import logging
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class VLLMServer:
    cmd: list[str]

    def __post_init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.port = int(
            [part for part in self.cmd if "--port" in part][0].split("=")[1]
        )

    def start(self) -> None:
        """
        Start the vLLM server and wait for it to be ready.
        Raises RuntimeError if the server process dies unexpectedly.
        Raises TimeoutError if the server fails to start after 10 minutes.
        """
        logger.info("Starting vLLM server with command: %s", " ".join(self.cmd))
        self.process = subprocess.Popen(self.cmd)

        # Wait for server to be ready
        max_attempts = 40  # 10 minutes total (40 * 15s = 600s = 10min)
        wait_time = 15  # seconds between attempts

        for attempt in range(max_attempts):
            if not self.is_running():
                raise RuntimeError("Server process died unexpectedly")

            try:
                response = requests.get(f"http://localhost:{self.port}/v1/models")
                if response.status_code == 200:
                    logger.info("Server is ready!")
                    return
            except requests.exceptions.ConnectionError:
                pass

            logger.info(
                "Attempt %d/%d: Server not ready yet, waiting %ds... (%ds elapsed)",
                attempt + 1,
                max_attempts,
                wait_time,
                (attempt + 1) * wait_time,
            )
            time.sleep(wait_time)

        raise TimeoutError(
            f"Server failed to start after {max_attempts} attempts (10min timeout)"
        )

    def is_running(self) -> bool:
        """Check if the server process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def stop(self):
        """Stop the vLLM server."""
        if self.process is not None and self.is_running():
            logger.info("Stopping vLLM server (PID: %s)...", self.process.pid)
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not terminate gracefully, forcing...")
                self.process.kill()
            self.process = None

    def __enter__(self):
        """Context manager entry point."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.stop()


@contextmanager
def spinup_vllm_server(no_op: bool, vllm_command: list[str]):
    if no_op:
        yield
        return

    server = VLLMServer(cmd=vllm_command)
    try:
        server.start()
        yield server
    finally:
        server.stop()
