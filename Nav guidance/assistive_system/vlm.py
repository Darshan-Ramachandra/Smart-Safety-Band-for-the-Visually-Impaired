import json
import urllib.error
import urllib.request


class LocalVLMReasoner:
    def __init__(self, ollama_url: str, model_name: str):
        self.url = ollama_url.rstrip("/") + "/api/generate"
        self.model_name = model_name

    def summarize(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return str(body.get("response", "")).strip()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return ""

