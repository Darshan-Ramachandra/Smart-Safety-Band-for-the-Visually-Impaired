import os

import cv2

try:
    from config import GEMINI_API_KEY as CONFIG_GEMINI_API_KEY
    from config import GEMINI_API_KEYS as CONFIG_GEMINI_API_KEYS
except ImportError:
    CONFIG_GEMINI_API_KEY = ""
    CONFIG_GEMINI_API_KEYS = []

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


BASE_PROMPT = (
    "Analyze this image for a visually impaired person.\n"
    "Describe:\n"
    "1. Environment (indoor/outdoor)\n"
    "2. Objects present\n"
    "3. Their relative positions (left/right/front)\n"
    "4. Any movement or danger\n\n"
    "Respond in SHORT sentences only. No paragraphs."
)


class ModelInference:
    MODELS = {
        "gemini": "gemini-2.5-flash",
    }

    def __init__(self, mode="gemini", google_api_key=None):
        self.mode = mode
        self.google_api_keys = self._resolve_google_api_keys(google_api_key)
        self.model_name = self.MODELS.get(mode)
        if self.model_name is None:
            raise ValueError(f"Unsupported MODEL mode: {mode}")

    def infer(self, image):
        if self.mode == "gemini":
            return self._infer_gemini(image)
        raise ValueError(f"Unknown inference mode: {self.mode}")

    def _resolve_google_api_keys(self, single_key):
        env_keys_raw = os.getenv("GOOGLE_API_KEYS", "")
        env_keys = [key.strip() for key in env_keys_raw.split(",") if key.strip()]
        config_keys = [str(key).strip() for key in CONFIG_GEMINI_API_KEYS if str(key).strip()]

        keys = []
        if single_key and single_key.strip():
            keys.append(single_key.strip())
        if os.getenv("GOOGLE_API_KEY", "").strip():
            keys.append(os.getenv("GOOGLE_API_KEY").strip())
        if CONFIG_GEMINI_API_KEY and CONFIG_GEMINI_API_KEY.strip():
            keys.append(CONFIG_GEMINI_API_KEY.strip())
        keys.extend(env_keys)
        keys.extend(config_keys)

        unique_keys = []
        seen = set()
        for key in keys:
            if key not in seen:
                seen.add(key)
                unique_keys.append(key)
        return unique_keys

    def _infer_gemini(self, image):
        if not self.google_api_keys:
            raise RuntimeError(
                "Gemini API key is required. "
                "Set `GEMINI_API_KEYS` in config.py, set `$env:GOOGLE_API_KEYS` "
                "(comma-separated), set `$env:GOOGLE_API_KEY`, or pass `--google-api-key`."
            )
        if genai is None or genai_types is None:
            raise RuntimeError(
                "The `google-genai` package is required for Gemini mode. "
                "Install it with `pip install google-genai`."
            )

        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Unable to encode image for Gemini inference.")

        image_bytes = encoded.tobytes()

        errors = []
        for api_key in self.google_api_keys:
            try:
                with genai.Client(api_key=api_key) as client:
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=[
                            BASE_PROMPT,
                            genai_types.Part.from_bytes(
                                data=image_bytes,
                                mime_type="image/png",
                            ),
                        ],
                        config=genai_types.GenerateContentConfig(
                            temperature=0.2,
                        ),
                    )
                return self._extract_gemini_text(response)
            except Exception as exc:
                errors.append(str(exc))

        raise RuntimeError(
            "Gemini inference failed for all configured API keys. "
            f"Last error: {errors[-1] if errors else 'unknown error'}"
        )

    def _extract_gemini_text(self, response):
        if response is None:
            return ""
        if isinstance(response, str):
            return response.strip()
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        if isinstance(response, dict):
            if "output_text" in response and response["output_text"]:
                return str(response["output_text"]).strip()
            if "text" in response and response["text"]:
                return str(response["text"]).strip()
            if "output" in response and response["output"]:
                return self._extract_gemini_from_output(response["output"])
        if hasattr(response, "output") and response.output:
            return self._extract_gemini_from_output(response.output)
        return ""

    def _extract_gemini_from_output(self, output):
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, list):
            parts = []
            for item in output:
                if isinstance(item, dict):
                    content = item.get("content")
                    if isinstance(content, list):
                        for sub in content:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                parts.append(str(sub.get("text", "")))
                    elif isinstance(content, str):
                        parts.append(content)
                elif isinstance(item, str):
                    parts.append(item)
            return " ".join(p.strip() for p in parts if p).strip()
        return ""
