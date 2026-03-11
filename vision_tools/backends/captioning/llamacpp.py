"""
LlamaCppBackend — Captioning via local llama.cpp server.

Ported from ``LlamaCppCaptioner``.
"""
from __future__ import annotations

import atexit
import logging
import os
import shutil
import socket
import subprocess
import time
from typing import Any

import cv2
import numpy as np
import requests

from vision_tools.backends.registry import BackendRegistry
from vision_tools.core.graph_types import Caption
from vision_tools.core.node import NodeContext

logger = logging.getLogger(__name__)


@BackendRegistry.register(task="captioning", model="llamacpp")
class LlamaCppBackend:
    """Backend for llama.cpp server-based VLM captioning.

    Automatically manages a local llama-server process.
    Scales GPU layers based on available VRAM.
    """

    def __init__(self) -> None:
        self._server_process = None
        self._server_url = None
        self._port = None
        self._model_id = ""
        self._imgsz = 512

    def load_model(self, model_path: str, device: str = "auto") -> Any:
        """Start llama-server and return the process handle."""
        self._model_id = model_path

        model_locator = "m" if os.path.exists(model_path) else "hf"

        self._port = self._find_free_port()
        self._server_url = f"http://127.0.0.1:{self._port}"

        server_path = os.environ.get("LLAMA_SERVER_PATH") or shutil.which("llama-server")
        if not server_path:
            server_path = "/home/linuxbrew/.linuxbrew/bin/llama-server"
            if not os.path.exists(server_path):
                raise RuntimeError(
                    "llama-server not found. Install llama.cpp or set LLAMA_SERVER_PATH."
                )

        gpu_layers = self._calculate_gpu_layers(device)
        cmd = [
            server_path,
            f"-{model_locator}", model_path,
            "--port", str(self._port),
            "--n-gpu-layers", str(gpu_layers),
            "-c", "8192",
            "--jinja",
        ]

        logger.info(f"LlamaCppBackend: starting server on port {self._port}")
        self._server_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        atexit.register(self.shutdown)
        self._wait_for_server()

        return self._server_process

    def infer(self, model: Any, inputs: Any) -> Any:
        """Send chat completion request to llama-server."""
        try:
            response = requests.post(
                f"{self._server_url}/v1/chat/completions",
                json=inputs,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"LlamaCppBackend: inference failed: {e}")
            return {}

    def postprocess(self, raw_output: Any, context: NodeContext) -> dict:
        """Extract caption to the canonical caption payload."""
        content = raw_output.get("choices", [{}])[0].get("message", {}).get("content", "")
        caption = Caption(text=content, model_id=self._model_id)
        return {"caption": caption}

    def preprocess(self, inputs: np.ndarray) -> dict:
        """Prepare frame as chat completion payload."""
        from vision_tools.utils.image_utils import base64_encode

        h, w = inputs.shape[:2]
        if max(h, w) > self._imgsz:
            scale = self._imgsz / max(h, w)
            inputs = cv2.resize(inputs, (int(w * scale), int(h * scale)))

        base64_image = base64_encode(inputs, "jpeg")
        return {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }],
            "max_tokens": 32,
        }

    def shutdown(self) -> None:
        """Stop llama-server process."""
        if self._server_process:
            logger.info("LlamaCppBackend: stopping server")
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            self._server_process = None

    def get_available_runtimes(self) -> list[str]:
        return ["cpu", "cuda"]

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @staticmethod
    def _calculate_gpu_layers(device: str) -> int:
        if device not in ("cuda", "auto"):
            return 0
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / (1024 ** 3)
                if vram_gb >= 16:
                    return 100
                elif vram_gb >= 8:
                    return 40
                elif vram_gb >= 4:
                    return 20
        except ImportError:
            pass
        return 0

    def _wait_for_server(self, timeout: int = 60) -> None:
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"{self._server_url}/health")
                if resp.status_code == 200:
                    logger.info("LlamaCppBackend: server ready")
                    return
            except requests.ConnectionError:
                pass

            if self._server_process.poll() is not None:
                stdout, stderr = self._server_process.communicate()
                raise RuntimeError(
                    f"llama-server failed.\nStdout: {stdout.decode()}\nStderr: {stderr.decode()}"
                )
            time.sleep(0.5)
        raise RuntimeError("Timeout waiting for llama-server")
