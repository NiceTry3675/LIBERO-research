"""The Clef client's image guard (branchlab.clef): no network or key."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from branchlab import clef as clef_mod

IMAGE = "data:image/png;base64,AAAA"
QUESTION = {"q": {"type": "noul", "instructions": "Is it there?"}}


def env_file():
    f = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
    f.write("CLOUDFLARE_AUTH_TOKEN=t\nCLOUDFLARE_ACCOUNT_ID=a\nOPENROUTER_API_KEY=k\n")
    f.close()
    return f.name


class FakeWorkers:
    """Stands in for urlopen: answers the colour probe from the image it carries (or wrongly, if blind)."""
    def __init__(self, blind=False):
        self.blind, self.bodies = blind, []

    def __call__(self, request, timeout=None):
        body = json.loads(request.data)
        self.bodies.append(body)
        if "colour" in body["questions"]:
            import base64
            from PIL import Image
            png = base64.b64decode(body["images"][0].split(",", 1)[1])
            red = Image.open(io.BytesIO(png)).getpixel((0, 0))[0] > 128
            seen = "red" if self.blind or red else "blue"      # a blind model answers the same for both
            answers = {"colour": {"probabilities": {"red": 0.9 if seen == "red" else 0.1,
                                                    "blue": 0.1 if seen == "red" else 0.9}}}
        else:
            answers = {"q": {"noul": 0.7}}
        return io.BytesIO(json.dumps({"result": {"answers": answers}}).encode())


class ClefImageGuard(unittest.TestCase):
    def setUp(self):
        self.env = env_file()

    def test_default_backend_is_workers(self):
        self.assertEqual(clef_mod.Clef(env_path=self.env).backend, "workers")

    def test_openrouter_refuses_images_but_allows_text(self):
        client = clef_mod.Clef(env_path=self.env, backend="openrouter")
        with self.assertRaises(ValueError):
            client.ask("state", QUESTION, images=[IMAGE])

    def test_probe_runs_once_before_the_first_image_call(self):
        fake = FakeWorkers()
        client = clef_mod.Clef(env_path=self.env)
        with patch("urllib.request.urlopen", fake):
            client.ask("state", QUESTION, images=[IMAGE])
            client.ask("state", QUESTION, images=[IMAGE])
            client.ask("state", QUESTION)
        kinds = ["colour" if "colour" in b["questions"] else "q" for b in fake.bodies]
        self.assertEqual(kinds, ["colour", "colour", "q", "q", "q"])

    def test_blind_backend_raises(self):
        client = clef_mod.Clef(env_path=self.env)
        with patch("urllib.request.urlopen", FakeWorkers(blind=True)), self.assertRaises(RuntimeError):
            client.ask("state", QUESTION, images=[IMAGE])
        self.assertFalse(client._images_ok)

    def test_cached_answers_skip_the_probe(self):
        cache = Path(tempfile.mkdtemp())/"c.jsonl"
        cache.write_text(json.dumps({"key": "k", "result": {"answers": {"q": {"noul": 0.2}}}}) + "\n")
        client = clef_mod.Clef(env_path=self.env, cache_path=cache, backend="openrouter")
        self.assertEqual(client.ask("state", QUESTION, images=[IMAGE], key="k")["answers"]["q"]["noul"], 0.2)


if __name__ == "__main__":
    unittest.main()
