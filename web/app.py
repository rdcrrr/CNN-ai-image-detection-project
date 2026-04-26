"""
app.py
------
Flask API server for the AI Image Detection system.
Supports two image types: "general" and "face".

Place at:  imageMetadata/web/app.py

Run with:
  cd imageMetadata
  pip install flask flask-cors
  python web/app.py
"""

from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Path setup — project root is one level above web/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CNN.feature_extract.inference    import run_inference, load_all_models
from CNN.feature_extract.score_fusion import fuse_scores

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    # ── Validate upload ───────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not _allowed(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # ── Get image type from form data (default: general) ──────────────────
    image_type = request.form.get("image_type", "general")
    if image_type not in ("general", "face"):
        image_type = "general"

    # ── Save to temp file ─────────────────────────────────────────────────
    suffix   = Path(file.filename).suffix.lower()
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # ── Run inference ─────────────────────────────────────────────────
        raw = run_inference(tmp_path, image_type=image_type)

        # ── Fuse scores ───────────────────────────────────────────────────
        fusion = fuse_scores(
            prnu_score      = raw["prnu_score"],
            ela_score       = raw["ela_score"],
            freq_score      = raw["freq_score"],
            metadata_score  = raw["metadata_score"],
            metadata_format = raw["metadata_format"],
            metadata_reason = raw["metadata_reason"],
            image_type      = image_type,
        )

        return jsonify({
            "verdict":    fusion["verdict"],
            "confidence": fusion["confidence"],
            "score":      fusion["final_score"],
            "image_type": image_type,
        })

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading models...")
    load_all_models()
    print("Flask API running at http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)