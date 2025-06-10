from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import time
import requests
import logging
from typing import Optional
from contextlib import contextmanager
import concurrent.futures
import os

logger = logging.getLogger(__name__)


@dataclass
class VLLMServer:
    model_id: str
    port: int
    enable_reasoning: bool
    process: Optional[subprocess.Popen] = None
    rank: int = 0

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
            "--max-num-seqs=1000",  # throttling is done client-side
            "--gpu-memory-utilization=0.95",
            f"--port={self.port}",
        ]

        if self.enable_reasoning:
            cmd.extend(["--enable-reasoning", "--reasoning-parser=deepseek_r1"])
        else:
            chat_template = "qwen3_nonthinking.jinja"
            abs_fpath = (Path(__file__).parent / chat_template).absolute()
            cmd.extend([f"--chat-template={str(abs_fpath)}"])

        # Set up environment variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.rank)

        logger.info(
            "Starting vLLM server on port %s with command: %s (CUDA_VISIBLE_DEVICES=%s)", 
            self.port, " ".join(cmd), self.rank
        )
        self.process = subprocess.Popen(cmd, env=env)

        # Wait for server to be ready
        max_attempts = 40  # 10 minutes total (40 * 15s = 600s = 10min)
        wait_time = 15  # seconds between attempts

        for attempt in range(max_attempts):
            if not self.is_running():
                raise RuntimeError(
                    f"Server process on port {self.port} died unexpectedly"
                )

            try:
                response = requests.get(f"http://localhost:{self.port}/v1/models")
                if response.status_code == 200:
                    logger.info("Server on port %s is ready!", self.port)
                    return
            except requests.exceptions.ConnectionError:
                pass

            logger.info(
                "Port %s - Attempt %d/%d: Server not ready yet, waiting %ds... (%ds elapsed)",
                self.port,
                attempt + 1,
                max_attempts,
                wait_time,
                (attempt + 1) * wait_time,
            )
            time.sleep(wait_time)

        raise TimeoutError(
            f"Server on port {self.port} failed to start after {max_attempts} attempts (10min timeout)"
        )

    def is_running(self) -> bool:
        """Check if the server process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def stop(self):
        """Stop the vLLM server."""
        if self.process is not None and self.is_running():
            logger.info(
                "Stopping vLLM server on port %s (PID: %s)...",
                self.port,
                self.process.pid,
            )
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Server on port %s did not terminate gracefully, forcing...",
                    self.port,
                )
                self.process.kill()
            self.process = None

    def __enter__(self):
        """Context manager entry point."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.stop()


@dataclass
class MultiVLLMServerManager:
    """Manager for multiple vLLM servers."""

    model_id: str
    ports: list[int]
    enable_reasoning: bool
    servers: list[VLLMServer] = field(default_factory=list)

    def start(self) -> None:
        """Start all vLLM servers in parallel and wait for all to be ready."""
        if not self.ports:
            logger.info("No ports specified, not starting any servers")
            return

        logger.info(
            "Starting %d vLLM servers on ports: %s", len(self.ports), self.ports
        )

        # Create server instances
        self.servers = [
            VLLMServer(self.model_id, port, self.enable_reasoning, rank=i)
            for i, port in enumerate(self.ports)
        ]

        # Start all servers in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.servers)
        ) as executor:
            futures = [executor.submit(server.start) for server in self.servers]

            # Wait for all servers to start successfully
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    future.result()  # This will raise any exception that occurred
                    logger.info(
                        "Server %d/%d started successfully", i + 1, len(self.servers)
                    )
                except Exception as e:
                    logger.error(
                        "Failed to start server on port %s: %s", self.servers[i].port, e
                    )
                    # Stop any servers that did start
                    self.stop()
                    raise

        logger.info("All %d vLLM servers are ready!", len(self.servers))

    def stop(self) -> None:
        """Stop all vLLM servers."""
        if not self.servers:
            return

        logger.info("Stopping %d vLLM servers...", len(self.servers))

        # Stop all servers in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.servers)
        ) as executor:
            futures = [executor.submit(server.stop) for server in self.servers]
            concurrent.futures.wait(futures)

        self.servers = []
        logger.info("All vLLM servers stopped")

    def __enter__(self):
        """Context manager entry point."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.stop()


@contextmanager
def spinup_vllm_servers(
    no_op: bool, model_id: str, ports: list[int], enable_reasoning: bool
):
    """
    Context manager for managing a multiple vLLM servers lifecycle.
    Example:
        with spinup_vllm_servers(..) as server:
            # Do inference here
            pass
    """
    if no_op:
        yield
        return

    server = MultiVLLMServerManager(model_id, ports, enable_reasoning)
    try:
        server.start()
        yield server
    finally:
        server.stop()
