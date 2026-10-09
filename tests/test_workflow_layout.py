"""Guard relocated CLI imports, repository paths and web subprocess targets."""
import ast
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / "workflows").glob("*/*.py"))
MOVED_NAMES = {path.stem for path in WORKFLOWS if path.name != "__init__.py"}


class WorkflowLayoutTest(unittest.TestCase):
    def test_workflow_modules_resolve(self):
        for path in WORKFLOWS:
            module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.util.find_spec(module))

    def test_no_legacy_imports_or_wrong_repository_roots(self):
        sources = WORKFLOWS + list((ROOT / "tests").glob("*.py")) + list((ROOT / "tactical_analysis").rglob("*.py"))
        for path in sources:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    self.assertNotIn(node.module, MOVED_NAMES, str(path))
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name, MOVED_NAMES, str(path))
            if path in WORKFLOWS:
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id in {"ROOT", "PROJECT_ROOT"}
                        for target in node.targets
                    ):
                        actual = eval(compile(ast.Expression(node.value), str(path), "eval"),
                                      {"Path": Path, "__file__": str(path), "__builtins__": {}})
                        self.assertEqual(actual, ROOT, str(path))

    def test_web_module_targets_exist(self):
        targets = set()
        for path in (ROOT / "web_preview/server").glob("*.js"):
            source = path.read_text()
            targets.update(re.findall(r'"(workflows\.[\w.]+)"', source))
            for old in MOVED_NAMES:
                self.assertNotRegex(source, r'path\.join\((?:root|repoRoot),\s*"' + old + r'\.py"')
        self.assertGreaterEqual(len(targets), 9)
        for target in targets:
            self.assertTrue((ROOT / (target.replace(".", "/") + ".py")).is_file(), target)

    def test_module_command_help(self):
        for module in ("workflows.tactical.run_tactical_analysis", "workflows.review.run_openai_tactical_review",
                       "workflows.players.export_player_focus"):
            with self.subTest(module=module):
                result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                                        capture_output=True, text=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())

    def test_yolox_preprocessing_is_not_ignored(self):
        source = ROOT / "yolox/data/data_augment.py"
        self.assertTrue(source.is_file())
        # A source snapshot without .git must still run the suite.
        if (ROOT / ".git").exists():
            result = subprocess.run(["git", "check-ignore", str(source)], cwd=ROOT, capture_output=True)
            self.assertEqual(result.returncode, 1, result.stdout.decode())


if __name__ == "__main__":
    unittest.main()
