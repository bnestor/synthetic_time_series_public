import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class ImportGraphTests(unittest.TestCase):
    def test_training_does_not_import_evaluation(self):
        for path in (ROOT / "training").glob("*.py"):
            self.assertFalse(
                any(module.startswith("synthetic_time_series_clean.evaluation") for module in imported_modules(path)),
                path,
            )

    def test_evaluation_does_not_import_training(self):
        for path in (ROOT / "evaluation").glob("*.py"):
            self.assertFalse(
                any(module.startswith("synthetic_time_series_clean.training") for module in imported_modules(path)),
                path,
            )


if __name__ == "__main__":
    unittest.main()
