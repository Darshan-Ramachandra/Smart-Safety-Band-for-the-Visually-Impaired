import re


class SceneInterpreter:
    ENVIRONMENT_KEYWORDS = [
        "indoor",
        "outdoor",
        "road",
        "room",
        "hallway",
        "street",
        "park",
        "office",
        "kitchen",
        "living room",
        "bedroom",
    ]
    OBJECT_KEYWORDS = [
        "person",
        "people",
        "chair",
        "table",
        "vehicle",
        "car",
        "truck",
        "bus",
        "animal",
        "dog",
        "cat",
        "obstacle",
        "door",
        "stairs",
        "wall",
        "bicycle",
        "motorcycle",
    ]
    MOVEMENT_KEYWORDS = [
        "approaching",
        "moving away",
        "stationary",
        "still",
        "walking",
        "running",
        "standing",
        "coming",
        "going",
        "heading",
    ]
    POSITION_KEYWORDS = [
        "left",
        "right",
        "center",
        "front",
        "behind",
        "near",
        "far",
        "ahead",
        "in front",
        "on the left",
        "on the right",
    ]

    def __init__(self):
        self.last_announced = []

    def interpret(self, raw_text):
        if not raw_text:
            return []
        sentences = self._split_sentences(raw_text)
        candidates = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if self._is_noise(sentence):
                continue
            if self._is_relevant(sentence):
                normalized = self._normalize(sentence)
                if normalized and normalized not in candidates:
                    candidates.append(normalized)
        output = self._filter_duplicates(candidates)
        return output

    def _split_sentences(self, text):
        raw_sentences = re.split(r"(?<=[.!?])\\s+", text.strip())
        normalized = []
        for sentence in raw_sentences:
            sentence = sentence.replace("\n", " ").strip()
            if sentence:
                normalized.append(sentence.rstrip(".?!"))
        return normalized

    def _is_noise(self, sentence):
        lower = sentence.lower()
        noise_terms = [
            "cannot detect",
            "unable to",
            "no obvious",
            "maybe",
            "not sure",
            "unclear",
            "unknown",
        ]
        return any(term in lower for term in noise_terms)

    def _is_relevant(self, sentence):
        lower = sentence.lower()
        if any(keyword in lower for keyword in self.ENVIRONMENT_KEYWORDS):
            return True
        if any(keyword in lower for keyword in self.OBJECT_KEYWORDS):
            return True
        if any(keyword in lower for keyword in self.MOVEMENT_KEYWORDS):
            return True
        if any(keyword in lower for keyword in self.POSITION_KEYWORDS):
            return True
        return False

    def _normalize(self, sentence):
        sentence = re.sub(r"\s+", " ", sentence).strip()
        sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
        return sentence

    def _filter_duplicates(self, candidates):
        filtered = []
        for sentence in candidates:
            if sentence in self.last_announced:
                continue
            filtered.append(sentence)
        self.last_announced = filtered[-10:]
        return filtered
