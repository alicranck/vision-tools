import os
import subprocess
import time
import socket
import logging
import requests
import atexit
import shutil
from transformers import AutoProcessor, AutoTokenizer
from optimum.intel.openvino.modeling_visual_language import OVModelForVisualCausalLM, OVWeightQuantizationConfig
import numpy as np
import cv2

from .base_tool import BaseVisionTool, ToolKey
from ...utils.image_utils import base64_encode
from ...utils.types import ImageHandle, Any

logger = logging.getLogger(__name__)


current_dir = __file__.rsplit("/", 1)[0]
DEFAULT_IMAGE_SIZE = 512


class Captioner(BaseVisionTool): 
    """
    Captioning tool using SmolVLM2.
    """
    def __init__(self, model_id, config, device = 'cpu'):
        self.processor = None
        self.tokenizer = None
        super().__init__(model_id, config, device)

    def _load_model(self):

        model = OVModelForVisualCausalLM.from_pretrained(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        return model

    def preprocess(self, frame: np.ndarray) -> np.ndarray:

        messages = [
            {
                "role": "user",
                "content": [
                {"type": "text", "text": "Give a concise description of what is happening in this image"},
                {"type": "image", "url": f"data:image/png;base64,{base64_encode(frame, 'png')}"},
                ]
            },
        ]

        inputs = self.processor.apply_chat_template(messages, 
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    return_dict=True,
                                                    return_tensors="pt"
                                                    ).to(self.device)
        return inputs

    def inference(self, model_inputs: Any) -> Any:

        generated_ids = self.model.generate(**model_inputs, do_sample=False,
                                             max_new_tokens=64)
        generated_texts = self.processor.batch_decode(generated_ids,
                                                       skip_special_tokens=True)

        return generated_texts

    def postprocess(self, raw_output: Any, original_shape: tuple) -> dict:
        assitant_response = raw_output[0].split("Assistant:")[1].strip()
        data = {"caption": assitant_response}
        return data
    
    @property
    def output_keys(self) -> list:
        caption = ToolKey(
            key_name="caption",
            data_type=str,
            description="Generated captions describing the input image",
        )
        return [caption]

    @property
    def processing_input_keys(self) -> list:
        return []

    @property
    def config_keys(self) -> list:
        return []


class LlamaCppCaptioner(BaseVisionTool):
    """
    Captioning tool using a local llama.cpp server.
    """
    def __init__(self, model_id, config, device='cpu'):
        self.server_process = None
        self.port = None
        self.server_url = None
        self.imgsz: int
        super().__init__(model_id, config, device)

    def _configure(self, config: dict):
        self.imgsz = config.get('imgsz', DEFAULT_IMAGE_SIZE)
        return

    def _find_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def _load_model(self):
        model_path = self.model_id
        if os.path.exists(model_path):
            model_locator = 'm'
        else:
            model_locator = 'hf'

        self.port = self._find_free_port()
        self.server_url = f"http://127.0.0.1:{self.port}"
        
        server_path = os.environ.get("LLAMA_SERVER_PATH") or shutil.which("llama-server")
        if not server_path:
             # Fallback to the old hardcoded path for local dev if not found
             server_path = "/home/linuxbrew/.linuxbrew/bin/llama-server"
             if not os.path.exists(server_path):
                 raise RuntimeError("llama-server executable not found. Please install llama.cpp or set LLAMA_SERVER_PATH.")

        cmd = [
            server_path,
            f"-{model_locator}", model_path,
            "--port", str(self.port),
            "--n-gpu-layers", "100" if self.device == "cuda" else "0",
            "-c", "8192",
            "--jinja"
        ]
        
        logger.info(f"Starting llama-server on port {self.port}...")
        self.server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Register cleanup
        atexit.register(self.unload_tool)
        
        # Wait for server to be ready
        self._wait_for_server()
        
        return self.server_process

    def _wait_for_server(self, timeout=60):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/health")
                if response.status_code == 200:
                    logger.info("llama-server is ready.")
                    return
            except requests.ConnectionError:
                pass
            
            if self.server_process.poll() is not None:
                stdout, stderr = self.server_process.communicate()
                raise RuntimeError(f"llama-server failed to start.\nStdout: {stdout.decode()}\nStderr: {stderr.decode()}")
            
            time.sleep(0.5)
        raise RuntimeError("Timeout waiting for llama-server to start.")

    def unload_tool(self):
        if self.server_process:
            logger.info("Stopping llama-server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
        super().unload_tool()

    def preprocess(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        if max(h, w) > self.imgsz:
            scale = self.imgsz / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        base64_image = base64_encode(frame, 'jpeg')
        
        payload = {"messages": [
                {"role": "user", "content": 
                    [{"type": "text", "text": "Describe the image"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
                "max_tokens": 32}
        return payload

    def inference(self, model_inputs: Any) -> Any:
        try:
            response = requests.post(f"{self.server_url}/v1/chat/completions", 
                                     json=model_inputs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {}

    def postprocess(self, raw_output: Any, original_shape: tuple) -> dict:
        content = raw_output.get("choices")[0].get("message").get("content")
        return {"caption": content}

    @property
    def output_keys(self) -> list:
        return [
            ToolKey("caption", str, "Generated caption from llama.cpp")
        ]

    @property
    def processing_input_keys(self) -> list:
        return []

    @property
    def config_keys(self) -> list:
        return []

