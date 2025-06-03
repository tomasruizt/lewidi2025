from pathlib import Path
import subprocess
import time
import requests
import logging
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class VLLMServer:
    def __init__(self, model_id: str, port: int, use_reasoning_args: bool):
        self.model_id = model_id
        self.port = port
        self.use_reasoning_args = use_reasoning_args
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        """
        Start the vLLM server and wait for it to be ready.
        Raises RuntimeError if the server process dies unexpectedly.
        Raises TimeoutError if the server fails to start after 10 minutes.
        """
        cmd = [
            "vllm",
            "serve",
            self.model_id,
            "--task=generate",
            "--disable-log-requests",  # prevents logging the prompt
            "--disable-uvicorn-access-log",  # prevents logging 200 OKs
            "--max-model-len=16k",
            "--gpu-memory-utilization=0.95",
            f"--port={self.port}",
        ]

        if self.use_reasoning_args:
            cmd.extend(["--enable-reasoning", "--reasoning-parser=deepseek_r1"])
        else:
            chat_template = "qwen3_nonthinking.jinja"
            abs_fpath = (Path(__file__).parent / chat_template).absolute()
            cmd.extend([f"--chat-template={str(abs_fpath)}"])

        logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd)

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
                f"Attempt {attempt + 1}/{max_attempts}: Server not ready yet, waiting {wait_time}s... ({(attempt + 1) * wait_time}s elapsed)"
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
            logger.info(f"Stopping vLLM server (PID: {self.process.pid})...")
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
def spinup_vllm_server(no_op: bool, model_id: str, port: int, use_reasoning_args: bool):
    """
    Context manager for managing a vLLM server lifecycle.

    Example:
        with spinup_vllm_server(..) as server:
            # Do inference here
            pass
    """
    if no_op:
        yield
        return

    server = VLLMServer(model_id, port, use_reasoning_args)
    try:
        server.start()
        yield server
    finally:
        server.stop()
