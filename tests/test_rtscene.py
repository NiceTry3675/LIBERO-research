"""Run-time rtscene modules never read the simulator's truth: only labels.py (scoring and calibration) does."""
import ast
from pathlib import Path
import unittest

PKG = Path(__file__).resolve().parents[1]/"src/branchlab/rtscene"
TRUTH_KEYS = {"objects", "success", "Segmentation", "replay_success"}


def strings(tree):
    """String constants other than docstrings."""
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
            and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


class NoTruthAtRunTime(unittest.TestCase):
    def test_only_labels_reads_truth(self):
        modules = sorted(p for p in PKG.glob("*.py") if p.name != "labels.py")
        self.assertIn("heightmap.py", {p.name for p in modules})
        for path in modules:
            tree = ast.parse(path.read_text())
            with self.subTest(module=path.name):
                imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
                imports |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
                self.assertFalse({"labels", "branchlab.rtscene.labels"} & imports, "imports labels")
                consts = strings(tree)
                self.assertFalse(TRUTH_KEYS & set(consts), f"reads {TRUTH_KEYS & set(consts)}")
                seg = [c for c in consts if c.endswith("_seg")]
                # capture.py names the segmentation arrays only to leave them out of Episode.arrays
                self.assertEqual(seg, ["_seg"] if path.name == "capture.py" else [])


if __name__ == "__main__":
    unittest.main()
