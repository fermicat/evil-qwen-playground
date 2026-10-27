import argparse


def main() -> None:
    """Start the playground on http://127.0.0.1:8792 (loads the model first, about 5 s)."""
    import uvicorn

    ap = argparse.ArgumentParser(description="Evil Qwen playground")
    ap.add_argument("--port", type=int, default=8792)
    a = ap.parse_args()
    uvicorn.run("evil_qwen_playground.server:app", host="127.0.0.1", port=a.port)
