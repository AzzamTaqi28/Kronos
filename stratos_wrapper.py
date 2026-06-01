#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

TOKENIZER_REPO = os.getenv("KRONOS_TOKENIZER_REPO", "NeoQuasar/Kronos-Tokenizer-base")
MODEL_REPO = os.getenv("KRONOS_MODEL_REPO", "NeoQuasar/Kronos-small")
TOKENIZER_REVISION = os.getenv("KRONOS_TOKENIZER_REVISION", "0e0117387f39004a9016484a186a908917e22426")
MODEL_REVISION = os.getenv("KRONOS_MODEL_REVISION", "901c26c1332695a2a8f243eb2f37243a37bea320")
MAX_CONTEXT = int(os.getenv("KRONOS_MAX_CONTEXT", "512"))
DEVICE = os.getenv("KRONOS_DEVICE") or None

TIMEFRAME_TO_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000
}


def timeframe_to_ms(timeframe: str) -> int:
    return TIMEFRAME_TO_MS.get(timeframe, 60 * 60_000)


def load_payload() -> dict:
    try:
        return json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON payload: {exc}") from exc


def build_prediction_timestamps(last_ts: pd.Timestamp, timeframe: str, horizon: int) -> pd.DatetimeIndex:
    step_ms = timeframe_to_ms(timeframe)
    start = last_ts + pd.to_timedelta(step_ms, unit="ms")
    return pd.date_range(start=start, periods=horizon, freq=pd.to_timedelta(step_ms, unit="ms"))


def build_frame(payload: dict):
    candles = payload.get("candles", [])
    if not candles:
        raise SystemExit("Payload must include candles")

    df = pd.DataFrame(candles)
    for column in ("open", "high", "low", "close"):
        if column not in df.columns:
            raise SystemExit(f"Missing required candle column: {column}")
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    else:
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="raise")
    df = df.sort_values("ts").reset_index(drop=True)

    timeframe = str(payload.get("timeframe", "1h"))
    horizon = int(payload.get("horizonCandles", 12))
    x_timestamp = df["ts"].reset_index(drop=True)
    y_timestamp = pd.Series(build_prediction_timestamps(x_timestamp.iloc[-1], timeframe, horizon))
    return df, x_timestamp, y_timestamp, timeframe, horizon


def summarize_input(df: pd.DataFrame) -> dict:
    closes = df["close"].astype(float)
    return {
        "candleCount": int(len(df)),
        "firstTs": df["ts"].iloc[0].isoformat(),
        "lastTs": df["ts"].iloc[-1].isoformat(),
        "lastClose": float(closes.iloc[-1]),
        "highLowRangePct": float((((df["high"].max() - df["low"].min()) / max(closes.iloc[-1], 1e-9)) * 100).round(4)),
        "volatilityPct": float(closes.pct_change().fillna(0).std() * 100 if len(closes) > 1 else 0.0),
    }


def infer_direction(pred_df: pd.DataFrame, last_close: float) -> str:
    predicted_close = float(pred_df["close"].iloc[-1])
    return "long" if predicted_close >= last_close else "short"


def infer_confidence(pred_df: pd.DataFrame, last_close: float) -> float:
    predicted_close = float(pred_df["close"].iloc[-1])
    change_pct = 0.0 if last_close == 0 else abs((predicted_close - last_close) / last_close) * 100
    confidence = 55.0 + min(change_pct * 7.5, 35.0)
    return round(max(50.0, min(confidence, 95.0)), 2)


def main() -> None:
    payload = load_payload()
    df, x_timestamp, y_timestamp, timeframe, horizon = build_frame(payload)

    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_REPO, revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)

    pred_df = predictor.predict(
        df=df[["open", "high", "low", "close", "volume", "amount"]],
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=horizon,
        T=float(payload.get("temperature", 1.0)),
        top_k=int(payload.get("top_k", 0)),
        top_p=float(payload.get("top_p", 0.9)),
        sample_count=int(payload.get("sample_count", 1)),
        verbose=bool(payload.get("verbose", False)),
    )

    last_close = float(df["close"].iloc[-1])
    output = {
        "provider": "kronos-official-python",
        "modelVersion": f"{MODEL_REPO}@{MODEL_REVISION}",
        "runtime": {
            "mode": "process",
            "provider": "kronos-official-python",
            "modelVersion": f"{MODEL_REPO}@{MODEL_REVISION}",
        },
        "inputSummary": summarize_input(df),
        "direction": infer_direction(pred_df, last_close),
        "confidence": infer_confidence(pred_df, last_close),
        "reasonCodes": [
            "kronos_official",
            "python_adapter",
            "ohlcv_input"
        ],
        "forecastPoints": [
            {
                "horizonIndex": index + 1,
                "targetTs": pd.Timestamp(timestamp).isoformat(),
                "predictedOpen": float(row["open"]),
                "predictedHigh": float(row["high"]),
                "predictedLow": float(row["low"]),
                "predictedClose": float(row["close"]),
                "predictedVolume": float(row["volume"]),
                "predictedAmount": float(row["amount"]),
                "confidence": infer_confidence(pred_df.iloc[: index + 1], last_close),
            }
            for index, (timestamp, row) in enumerate(pred_df.iterrows())
        ],
        "notes": {
            "asset": payload.get("asset", {}).get("symbol"),
            "timeframe": timeframe,
        },
    }

    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Kronos wrapper failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
