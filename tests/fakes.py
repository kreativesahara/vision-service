"""Offline stand-ins for the Google clients."""
from types import SimpleNamespace


class FakeGemini:
    """Stands in for genai.Client: answers every request with a canned reply and records the requests."""

    def __init__(self, reply):
        self.reply = reply
        self.requests = []
        self.models = self

    def generate_content(self, **request):
        self.requests.append(request)
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(text=self.reply)


class FakeVision:
    """Stands in for vision.ImageAnnotatorClient, returning the same reply for every image."""

    def __init__(self, reply):
        self.reply = reply
        self.timeouts = []

    def text_detection(self, image, timeout=None):
        self.timeouts.append(timeout)
        return self.reply


def word(text, x, y, width=40, height=20):
    """One word annotation from Google Vision text detection, with its bounding box."""
    corners = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    return SimpleNamespace(
        description=text,
        bounding_poly=SimpleNamespace(vertices=[SimpleNamespace(x=cx, y=cy) for cx, cy in corners]),
    )


def vision_reply(*words, error=''):
    """A text_detection response: the full text first, then one annotation per word."""
    full_text = SimpleNamespace(description='\n'.join(w.description for w in words), bounding_poly=None)
    return SimpleNamespace(
        error=SimpleNamespace(message=error),
        text_annotations=[full_text, *words] if words else [],
    )
