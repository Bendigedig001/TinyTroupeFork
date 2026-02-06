import json
import subprocess
import sys
from pathlib import Path

import pytest

@pytest.mark.core
def test_main_mock_stdout_is_valid_json_with_image_options(tmp_path):
    image_module = pytest.importorskip("PIL.Image")

    repo_root = Path(__file__).resolve().parents[2]
    main_script = repo_root / "main.py"

    image_path = tmp_path / "option_a.png"
    image_module.new("RGB", (2, 2), color=(255, 255, 255)).save(image_path, "PNG")

    options_spec = {
        "question": "Pick your preferred drink based on labels and images.",
        "options": [
            {"id": "A", "label": "Tea", "images": [str(image_path)]},
            {"id": "B", "label": "Coffee", "images": []},
            {"id": "C", "label": "Hot Chocolate", "images": []},
        ],
    }

    options_path = tmp_path / "options.json"
    options_path.write_text(json.dumps(options_spec), encoding="utf-8")

    cache_dir = tmp_path / "cache"

    result = subprocess.run(
        [
            sys.executable,
            str(main_script),
            "--mock",
            "--quiet",
            "--no-persist",
            "--options-json",
            str(options_path),
            "--cache-dir",
            str(cache_dir),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    stdout = result.stdout.strip()
    assert stdout, "Expected JSON output on stdout."

    parsed = json.loads(stdout)
    assert isinstance(parsed, dict)
    assert "responses" in parsed
    assert len(parsed["responses"]) == 3
