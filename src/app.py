"""
Flask front end for the fraud detector. Serves the single-page UI, a JSON
scoring endpoint at /api/score/<address>, the live block stream, and the
Gemini-backed advisor chat.
"""
import os
import sys
import json
import time
import threading
from flask import Flask, jsonify, request, render_template_string, Response
import google.generativeai as genai

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from live_detector import FraudDetector
from adaptive_ensemble import scoring_config

app = Flask(__name__)

# Gemini key for the advisor chat
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

# small in-memory cache so the chat/feature endpoints don't re-hit Etherscan
# for an address we just scored
_score_cache = {}  # addr -> {"features": dict, "result": dict, "timestamp": float}
SCORE_CACHE_TTL = 600  # seconds

def _cache_score(address, features, result):
    _score_cache[address.lower()] = {
        "features": features,
        "result": result,
        "timestamp": time.time(),
    }

def _get_cached(address):
    entry = _score_cache.get(address.lower())
    if entry and (time.time() - entry["timestamp"]) < SCORE_CACHE_TTL:
        return entry
    return None

# load the models once, at import time
print("Initializing fraud detector...")
detector = FraudDetector()
print("Ready to serve requests!\n")

# ---------------------------------------------------------------------------
# Stream monitor (optional, only runs if an Ethereum provider is configured)
# ---------------------------------------------------------------------------
stream_monitor = None
stream_clients = []  # SSE client queues

def _broadcast(event_type, payload):
    """Push a named SSE event to all connected clients."""
    msg = f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
    dead = []
    for q in stream_clients:
        try:
            q.put_nowait(msg)
        except Exception:
            dead.append(q)
    for q in dead:
        stream_clients.remove(q)

def _broadcast_transaction(tx_data):
    """Fires the moment a tx shows up, before it's been scored."""
    _broadcast("transaction", tx_data)

def _broadcast_result(result):
    """Fires when fraud scoring completes for an address."""
    # Cache features for AI advisor, strip from SSE payload
    features = result.pop("features", None)
    if features:
        _cache_score(result.get("address", ""), features, result)
    _broadcast("score", result)

def _init_stream():
    """Try to initialise the stream monitor in the background."""
    global stream_monitor
    provider = (os.getenv("ETHEREUM_WS_URL")
                or os.getenv("ETHEREUM_HTTP_URL"))
    if not provider:
        print("Stream monitor: no Ethereum provider available — skipping.")
        return
    try:
        from stream_monitor import StreamMonitor
        stream_monitor = StreamMonitor(
            provider_uri=provider,
            fraud_threshold=0.7,
            max_addresses_per_block=4,
            on_result=_broadcast_result,
            on_transaction=_broadcast_transaction,
        )
        stream_monitor.start(blocking=False)
        print("Stream monitor started — listening for new blocks.")
    except Exception as e:
        print(f"Stream monitor failed to start: {e}")

# Launch in a thread so the Flask server isn't blocked during startup
threading.Thread(target=_init_stream, daemon=True).start()


# ============================================================================
# HTML template (manual search + live stream + detail panel, all one page)
# ============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Model for Fraud Detection in Blockchain Transactions</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #f5f6fa;
            --bg-card: #ffffff;
            --accent: #b1202e;
            --accent-light: rgba(177, 32, 46, 0.08);
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --success: #16a34a;
            --warning: #ca8a04;
            --danger: #dc2626;
        }

        [data-theme="dark"] {
            --bg: #0f172a;
            --bg-card: #1e293b;
            --accent-light: rgba(177, 32, 46, 0.18);
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --border: #334155;
            --success: #22c55e;
            --warning: #eab308;
            --danger: #ef4444;
        }
        [data-theme="dark"] .chat-bubble.ai { background: #334155; }
        [data-theme="dark"] input[type="text"] { background: #0f172a; }
        [data-theme="dark"] .stat-item { background: #0f172a; }
        [data-theme="dark"] .feed-header { background: #0f172a; }
        [data-theme="dark"] .tx-context { background: #0f172a; }
        [data-theme="dark"] .chat-messages { background: #0f172a; }
        [data-theme="dark"] .model-scores-list div:nth-child(odd) { background: #0f172a; }
        [data-theme="dark"] .list-panel-close { background: #334155; border-color: #475569; }
        [data-theme="dark"] .detail-close { background: #334155; border-color: #475569; }
        [data-theme="dark"] .empty-state .spinner { border-color: #334155; border-top-color: var(--accent); }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 0;
        }

        .uni-header {
            background: var(--accent);
            color: #fff;
            padding: 0.75rem 2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            font-size: 0.875rem;
            font-weight: 500;
            letter-spacing: 0.02em;
        }

        .uni-header .divider {
            width: 1px;
            height: 1.25rem;
            background: rgba(255,255,255,0.35);
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 2rem 2rem 3rem;
        }

        .page-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.25rem;
        }

        .page-subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-bottom: 2rem;
            line-height: 1.5;
        }

        .card {
            background: var(--bg-card);
            border-radius: 0.75rem;
            padding: 1.75rem;
            margin-bottom: 1.25rem;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }

        /* ── Search form ── */
        .search-form {
            display: flex;
            gap: 0.75rem;
        }
        .search-form label {
            display: block; font-size: 0.8rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.05em;
            color: var(--text-muted); margin-bottom: 0.5rem;
        }
        .input-wrap { flex: 1; }
        input[type="text"] {
            width: 100%; padding: 0.8rem 1rem; font-size: 0.95rem;
            background: var(--bg); border: 1.5px solid var(--border);
            border-radius: 0.5rem; color: var(--text);
            font-family: 'JetBrains Mono', monospace; transition: border-color 0.2s;
        }
        input[type="text"]:focus {
            outline: none; border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-light);
        }
        input[type="text"]::placeholder { color: #94a3b8; }
        .btn-wrap { display: flex; align-items: flex-end; }
        button {
            padding: 0.8rem 1.75rem; font-size: 0.95rem; font-weight: 600;
            background: var(--accent); border: none; border-radius: 0.5rem;
            color: white; cursor: pointer;
            transition: background 0.2s, box-shadow 0.2s; white-space: nowrap;
        }
        button:hover { background: #8f1a24; box-shadow: 0 4px 12px rgba(177,32,46,0.25); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }

        /* ── Shared result / detail styles ── */
        .section-label {
            font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.06em; color: var(--text-muted);
            margin-bottom: 0.75rem; padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }
        .result-header {
            display: flex; justify-content: space-between;
            align-items: center; flex-wrap: wrap; gap: 1rem;
        }
        .probability { font-size: 2.75rem; font-weight: 700; margin: 0.25rem 0; }
        .risk-badge {
            display: inline-block; padding: 0.4rem 1rem; border-radius: 2rem;
            font-weight: 700; font-size: 0.85rem; text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .risk-CRITICAL { background: rgba(220,38,38,0.1); color: #dc2626; border: 1.5px solid rgba(220,38,38,0.3); }
        .risk-HIGH     { background: rgba(234,88,12,0.1);  color: #ea580c; border: 1.5px solid rgba(234,88,12,0.3); }
        .risk-MEDIUM   { background: rgba(202,138,4,0.1);  color: #ca8a04; border: 1.5px solid rgba(202,138,4,0.3); }
        .risk-LOW      { background: rgba(22,163,74,0.1);  color: #16a34a; border: 1.5px solid rgba(22,163,74,0.3); }
        .risk-MINIMAL  { background: rgba(22,163,74,0.1);  color: #16a34a; border: 1.5px solid rgba(22,163,74,0.3); }
        .risk-UNKNOWN  { background: rgba(100,116,139,0.1);color: #64748b; border: 1.5px solid rgba(100,116,139,0.3); }

        .risk-gauge {
            width: 100%; height: 10px;
            background: linear-gradient(90deg, #16a34a, #ca8a04, #dc2626);
            border-radius: 5px; position: relative; margin: 1.25rem 0;
        }
        .risk-indicator {
            position: absolute; top: -5px; width: 20px; height: 20px;
            background: var(--bg-card); border: 3px solid var(--text);
            border-radius: 50%; transform: translateX(-50%);
            box-shadow: 0 1px 4px rgba(0,0,0,0.15); transition: left 0.5s ease-out;
        }
        .gauge-labels {
            display: flex; justify-content: space-between;
            font-size: 0.7rem; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.25rem;
        }

        .stats-grid {
            display: grid; grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem; margin-bottom: 1.5rem;
        }
        .stat-item { background: var(--bg); padding: 1rem; border-radius: 0.5rem; }
        .stat-label {
            font-size: 0.75rem; font-weight: 500; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.25rem;
        }
        .stat-value {
            font-size: 1.15rem; font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .model-scores-list {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem; line-height: 1.8;
        }
        .model-scores-list div {
            display: flex; justify-content: space-between;
            padding: 0.35rem 0.75rem; border-radius: 0.35rem;
        }
        .model-scores-list div:nth-child(odd) { background: var(--bg); }

        .loading { text-align: center; padding: 2rem; color: var(--text-muted); }
        .spinner {
            width: 36px; height: 36px; border: 3px solid var(--border);
            border-top-color: var(--accent); border-radius: 50%;
            animation: spin 0.8s linear infinite; margin: 0 auto 1rem;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .error {
            background: rgba(220,38,38,0.06); border: 1px solid rgba(220,38,38,0.2);
            color: #dc2626; padding: 1rem; border-radius: 0.5rem; font-size: 0.9rem;
        }

        .result { display: none; }
        .result.visible { display: block; animation: fadeIn 0.3s ease-out; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* ── Live stream section ── */
        .stream-section { margin-top: 2rem; }
        .stream-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 1rem;
        }
        .stream-title { font-size: 1.25rem; font-weight: 700; }
        .status-badge {
            display: inline-flex; align-items: center; gap: 0.4rem;
            font-size: 0.8rem; font-weight: 500; color: var(--text-muted);
        }
        .status-dot {
            width: 8px; height: 8px; border-radius: 50%;
            display: inline-block;
        }
        .status-dot.connected { background: var(--success); box-shadow: 0 0 6px var(--success); }
        .status-dot.disconnected { background: var(--danger); }

        .stats-bar {
            display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap;
        }
        .stat-box {
            background: var(--bg-card); border-radius: 0.5rem;
            padding: 0.75rem 1.25rem; border: 1px solid var(--border);
            flex: 1; min-width: 100px;
            cursor: pointer; transition: border-color 0.2s, box-shadow 0.2s;
        }
        .stat-box:hover { border-color: var(--accent); box-shadow: 0 2px 8px rgba(177,32,46,0.1); }
        .stat-box.active { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-light); }
        .stat-box .label {
            font-size: 0.65rem; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.05em;
        }
        .stat-box .value { font-size: 1.25rem; font-weight: 700; margin-top: 0.15rem; }

        /* ── Feed rows ── */
        .feed {
            display: flex; flex-direction: column; gap: 0;
            max-height: 480px; overflow-y: auto;
            border: 1px solid var(--border); border-radius: 0.75rem;
            background: var(--bg-card);
        }
        .feed-header {
            display: grid;
            grid-template-columns: 1fr 80px 80px 80px 100px;
            gap: 0.75rem; align-items: center;
            padding: 0.6rem 1.25rem;
            font-size: 0.7rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.05em;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
            background: var(--bg);
            position: sticky; top: 0; z-index: 2;
        }
        .tx-row {
            display: grid;
            grid-template-columns: 1fr 80px 80px 80px 100px;
            gap: 0.75rem; align-items: center;
            padding: 0.65rem 1.25rem;
            font-size: 0.875rem;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            transition: background 0.15s;
            border-left: 3px solid transparent;
        }
        .tx-row:hover { background: var(--accent-light); }
        .tx-row.selected { background: var(--accent-light); border-left-color: var(--accent); }
        .tx-row.alert { border-left-color: var(--danger); }
        .tx-row.alert:hover { background: rgba(220,38,38,0.06); }
        .tx-row.medium { border-left-color: var(--warning); }
        .tx-row.medium:hover { background: rgba(202,138,4,0.06); }
        .tx-row .address {
            font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            color: var(--text);
        }
        .tx-row .prob { font-weight: 600; text-align: right; }
        .risk-pill {
            display: inline-block; padding: 0.15rem 0.5rem; border-radius: 0.25rem;
            font-size: 0.7rem; font-weight: 600; text-align: center;
        }
        .time-cell { color: var(--text-muted); font-size: 0.75rem; text-align: right; }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-4px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .tx-row.new { animation: slideIn 0.25s ease-out; }

        .empty-state {
            text-align: center; color: var(--text-muted); padding: 3rem 1rem;
        }
        .empty-state .spinner {
            width: 32px; height: 32px;
            border: 3px solid var(--border);
            border-top-color: var(--accent); border-radius: 50%;
            animation: spin 0.8s linear infinite; margin: 0 auto 0.75rem;
        }

        /* ── List slide-out panel (left) ── */
        .list-overlay {
            display: none; position: fixed; inset: 0;
            background: rgba(0,0,0,0.3); z-index: 98;
        }
        .list-overlay.open { display: block; }
        .list-panel {
            position: fixed; top: 0; left: -420px; width: 420px;
            height: 100vh; background: var(--bg-card);
            box-shadow: 4px 0 24px rgba(0,0,0,0.12);
            z-index: 99; overflow-y: auto;
            transition: left 0.3s ease-out; padding: 0;
        }
        .list-panel.open { left: 0; }
        .list-panel-header {
            padding: 1.25rem 1.5rem; border-bottom: 1px solid var(--border);
            display: flex; justify-content: space-between; align-items: center;
            position: sticky; top: 0; background: var(--bg-card); z-index: 2;
        }
        .list-panel-header h3 { font-size: 1rem; font-weight: 700; margin: 0; }
        .list-panel-close {
            background: var(--bg); border: 1px solid var(--border);
            border-radius: 0.375rem; width: 32px; height: 32px;
            display: flex; align-items: center; justify-content: center;
            cursor: pointer; font-size: 1.1rem; color: var(--text-muted);
        }
        .list-panel-close:hover { background: var(--border); }
        .list-item {
            display: grid; grid-template-columns: 1fr 70px 80px;
            gap: 0.5rem; align-items: center;
            padding: 0.7rem 1.5rem; border-bottom: 1px solid var(--border);
            border-left: 3px solid transparent;
            cursor: pointer; transition: background 0.15s; font-size: 0.85rem;
        }
        .list-item:hover { background: var(--accent-light); }
        .list-item.risk-c-CRITICAL { border-left-color: #dc2626; }
        .list-item.risk-c-HIGH     { border-left-color: #ea580c; }
        .list-item.risk-c-MEDIUM   { border-left-color: #ca8a04; }
        .list-item.risk-c-LOW      { border-left-color: #16a34a; }
        .list-item.risk-c-MINIMAL  { border-left-color: #16a34a; }
        .list-item .li-addr {
            font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }
        .list-item .li-prob { font-weight: 600; text-align: right; }
        .list-empty {
            text-align: center; color: var(--text-muted); padding: 3rem 1.5rem; font-size: 0.9rem;
        }

        /* ── Detail slide-out panel ── */
        .detail-overlay {
            display: none; position: fixed; inset: 0;
            background: rgba(0,0,0,0.3); z-index: 100;
        }
        .detail-overlay.open { display: block; }
        .detail-panel {
            position: fixed; top: 0; right: -480px; width: 480px;
            height: 100vh; background: var(--bg-card);
            box-shadow: -4px 0 24px rgba(0,0,0,0.12);
            z-index: 101; overflow-y: auto;
            transition: right 0.3s ease-out;
            padding: 0;
        }
        .detail-panel.open { right: 0; }
        .detail-close {
            position: absolute; top: 1rem; right: 1rem;
            background: var(--bg); border: 1px solid var(--border);
            border-radius: 0.375rem; width: 32px; height: 32px;
            display: flex; align-items: center; justify-content: center;
            cursor: pointer; font-size: 1.1rem; color: var(--text-muted);
            transition: background 0.15s;
        }
        .detail-close:hover { background: var(--border); }
        .detail-body { padding: 1.5rem 1.75rem 2rem; }
        .detail-addr {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem; color: var(--text-muted);
            word-break: break-all; margin-bottom: 1.25rem;
            padding-bottom: 1rem; border-bottom: 1px solid var(--border);
        }
        .detail-addr a { color: var(--accent); text-decoration: none; }
        .detail-addr a:hover { text-decoration: underline; }
        .detail-section { margin-bottom: 1.5rem; }
        .detail-section .section-label { margin-top: 0; }

        .tx-context {
            background: var(--bg); border-radius: 0.5rem;
            padding: 0.85rem 1rem; font-size: 0.8rem;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.7;
        }
        .tx-context .ctx-label { color: var(--text-muted); }

        /* ── Footer ── */
        .footer {
            text-align: center; color: var(--text-muted);
            font-size: 0.8rem; margin-top: 2.5rem; line-height: 1.7;
        }
        .footer strong { color: var(--text); font-weight: 600; }
        .footer a { color: var(--accent); text-decoration: none; }
        .footer a:hover { text-decoration: underline; }
        .footer hr {
            border: none; border-top: 1px solid var(--border);
            margin: 0.75rem auto; width: 60px;
        }
        a { color: var(--accent); }

        /* ── AI Advisor Chat (embedded in detail panel) ── */
        .chat-toggle-btn {
            width: 100%; padding: 0.75rem; font-size: 0.95rem;
            background: linear-gradient(135deg, #1e293b, #334155);
            border: none; border-radius: 0.5rem; color: white;
            cursor: pointer; font-weight: 600; margin-top: 1.5rem;
            transition: background 0.2s, box-shadow 0.2s;
        }
        .chat-toggle-btn:hover {
            background: linear-gradient(135deg, #334155, #475569);
            box-shadow: 0 4px 12px rgba(30,41,59,0.3);
        }
        .chat-section {
            display: none; margin-top: 1rem;
            border-top: 1px solid var(--border); padding-top: 1rem;
        }
        .chat-section.open { display: flex; flex-direction: column; }
        .chat-section .section-label { margin-bottom: 0.5rem; }
        .chat-messages {
            max-height: 280px; min-height: 120px; overflow-y: auto;
            display: flex; flex-direction: column; gap: 0.5rem;
            padding: 0.5rem; background: var(--bg); border-radius: 0.5rem;
            margin-bottom: 0.75rem;
        }
        .chat-bubble {
            max-width: 88%; padding: 0.6rem 0.85rem; border-radius: 0.75rem;
            font-size: 0.85rem; line-height: 1.55; word-wrap: break-word;
        }
        .chat-bubble.ai {
            align-self: flex-start; background: #e2e8f0; color: var(--text);
            border-bottom-left-radius: 0.2rem;
        }
        .chat-bubble.user {
            align-self: flex-end; background: var(--accent); color: white;
            border-bottom-right-radius: 0.2rem;
        }
        .chat-input-bar {
            display: flex; gap: 0.5rem;
        }
        .chat-input-bar input[type="text"] {
            flex: 1; padding: 0.6rem 0.75rem; font-size: 0.85rem;
            font-family: 'Inter', sans-serif;
        }
        .chat-input-bar button {
            padding: 0.6rem 1.25rem; font-size: 0.85rem;
        }

        /* ── Toast notifications ── */
        .toast-container {
            position: fixed; bottom: 1.5rem; right: 1.5rem;
            display: flex; flex-direction: column-reverse; gap: 0.5rem;
            z-index: 200; max-width: 380px;
        }
        .toast {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 0.75rem; padding: 0.85rem 1.15rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
            display: flex; align-items: center; gap: 0.75rem;
            animation: toastIn 0.3s ease-out;
            border-left: 4px solid var(--danger); cursor: pointer;
        }
        .toast.medium { border-left-color: var(--warning); }
        .toast-icon { font-size: 1.25rem; flex-shrink: 0; }
        .toast-body { flex: 1; min-width: 0; }
        .toast-addr {
            font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
            color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }
        .toast-prob { font-weight: 700; font-size: 0.95rem; margin-top: 0.15rem; }
        .toast-dismiss {
            color: var(--text-muted); cursor: pointer; font-size: 1.1rem;
            padding: 0.25rem; flex-shrink: 0; background: none; border: none;
        }
        @keyframes toastIn {
            from { opacity: 0; transform: translateX(100px); }
            to { opacity: 1; transform: translateX(0); }
        }
        @keyframes toastOut {
            from { opacity: 1; transform: translateX(0); }
            to { opacity: 0; transform: translateX(100px); }
        }

        /* ── Copy to clipboard ── */
        .copy-btn {
            display: inline-flex; align-items: center; justify-content: center;
            width: 26px; height: 26px; border-radius: 0.25rem;
            background: none; border: 1px solid var(--border);
            color: var(--text-muted); cursor: pointer; font-size: 0.75rem;
            transition: all 0.15s; flex-shrink: 0; margin-left: 0.5rem; vertical-align: middle;
        }
        .copy-btn:hover { background: var(--accent-light); border-color: var(--accent); color: var(--accent); }
        .copy-btn.copied { background: rgba(22,163,74,0.1); border-color: var(--success); color: var(--success); }

        /* ── Model agreement ── */
        .model-agreement {
            display: flex; align-items: center; gap: 0.5rem;
            padding: 0.6rem 0.85rem; border-radius: 0.5rem;
            font-size: 0.8rem; font-weight: 600; margin-top: 0.75rem;
        }
        .model-agreement.strong { background: rgba(22,163,74,0.1); color: var(--success); }
        .model-agreement.partial { background: rgba(202,138,4,0.1); color: var(--warning); }
        .model-agreement.weak { background: rgba(220,38,38,0.1); color: var(--danger); }
        .agreement-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }

        /* ── Sparkline score bars ── */
        .score-bar-wrap {
            display: flex; align-items: center; gap: 0.5rem; flex: 1; max-width: 140px;
        }
        .score-bar-bg {
            flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden;
        }
        .score-bar-fill {
            height: 100%; border-radius: 3px; transition: width 0.5s ease-out;
        }
        .score-val { font-weight: 600; min-width: 52px; text-align: right; }

        /* ── Feature importance bars ── */
        .feature-bars { margin-top: 0.5rem; }
        .feature-bar-row {
            display: flex; align-items: center; gap: 0.5rem;
            padding: 0.35rem 0; font-size: 0.78rem;
        }
        .feature-bar-label {
            width: 160px; flex-shrink: 0; color: var(--text-muted);
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.72rem;
        }
        .feature-bar-track {
            flex: 1; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden;
        }
        .feature-bar-fill {
            height: 100%; border-radius: 4px; transition: width 0.6s ease-out; background: var(--accent);
        }
        .feature-bar-val {
            min-width: 55px; text-align: right;
            font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted);
        }

        /* ── Skeleton loaders ── */
        .skeleton {
            background: linear-gradient(90deg, var(--border) 25%, transparent 50%, var(--border) 75%);
            background-size: 200% 100%; animation: shimmer 1.5s infinite; border-radius: 0.375rem;
        }
        @keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
        .skeleton-line { height: 14px; margin-bottom: 0.5rem; }
        .skeleton-line.w-60 { width: 60%; }
        .skeleton-line.w-40 { width: 40%; }
        .skeleton-line.w-80 { width: 80%; }
        .skeleton-block { height: 48px; margin-bottom: 0.75rem; }
        .skeleton-block.big { height: 72px; }
        .skeleton-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem; margin-bottom: 1rem; }
        .skeleton-grid-item { height: 60px; border-radius: 0.5rem; }

        /* ── Secondary button ── */
        .btn-secondary {
            padding: 0.45rem 1rem; font-size: 0.8rem; font-weight: 600;
            background: var(--bg-card); border: 1.5px solid var(--border);
            border-radius: 0.5rem; color: var(--text); cursor: pointer; transition: all 0.2s;
        }
        .btn-secondary:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-light); }

        /* ── Keyboard shortcuts help ── */
        .shortcut-overlay {
            display: none; position: fixed; inset: 0;
            background: rgba(0,0,0,0.5); z-index: 300;
            align-items: center; justify-content: center;
        }
        .shortcut-overlay.open { display: flex; }
        .shortcut-modal {
            background: var(--bg-card); border-radius: 0.75rem;
            padding: 1.75rem; max-width: 420px; width: 90%;
            box-shadow: 0 16px 48px rgba(0,0,0,0.2);
        }
        .shortcut-modal h3 { margin-bottom: 1rem; font-size: 1rem; }
        .shortcut-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.4rem 0; border-bottom: 1px solid var(--border); font-size: 0.85rem;
        }
        .shortcut-row:last-child { border-bottom: none; }
        .shortcut-key {
            background: var(--bg); border: 1px solid var(--border);
            border-radius: 0.25rem; padding: 0.15rem 0.5rem;
            font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 600;
        }

        /* ── Adaptive weights display ── */
        .adaptive-weights {
            display: flex; gap: 0.5rem; margin-top: 0.5rem;
            padding: 0.5rem 0.75rem; background: var(--bg);
            border-radius: 0.35rem; font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
        }
        .adaptive-weights .aw-item {
            flex: 1; text-align: center; padding: 0.35rem;
            border-radius: 0.25rem;
        }
        .adaptive-weights .aw-label {
            font-size: 0.65rem; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.04em;
        }
        .adaptive-weights .aw-val {
            font-weight: 700; font-size: 0.95rem; margin-top: 0.15rem;
        }
        .mode-indicator {
            display: inline-flex; align-items: center; gap: 0.3rem;
            font-size: 0.72rem; padding: 0.2rem 0.6rem;
            border-radius: 1rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.04em;
        }
        .mode-indicator.fixed {
            background: rgba(99,102,241,0.1); color: #6366f1;
            border: 1px solid rgba(99,102,241,0.2);
        }
        .mode-indicator.adaptive {
            background: rgba(16,185,129,0.1); color: #10b981;
            border: 1px solid rgba(16,185,129,0.2);
        }
        .compare-btn {
            padding: 0.4rem 0.75rem; font-size: 0.78rem; font-weight: 600;
            background: none; border: 1.5px solid var(--accent); color: var(--accent);
            border-radius: 0.375rem; cursor: pointer; transition: all 0.2s;
        }
        .compare-btn:hover {
            background: var(--accent); color: white;
        }
        .compare-result {
            margin-top: 0.75rem; padding: 0.75rem;
            background: var(--bg); border-radius: 0.5rem;
            font-size: 0.85rem;
        }
        .compare-result table {
            width: 100%; border-collapse: collapse;
            font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
        }
        .compare-result th, .compare-result td {
            padding: 0.4rem 0.5rem; text-align: right;
            border-bottom: 1px solid var(--border);
        }
        .compare-result th { text-align: left; font-weight: 600; color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; }
        .compare-result td:first-child, .compare-result th:first-child { text-align: left; }
        #modeToggle.active-adaptive {
            background: rgba(16,185,129,0.2); border-color: #10b981;
        }

        /* ── Accessibility ── */
        :focus-visible {
            outline: 2px solid var(--accent); outline-offset: 2px;
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }

        /* ── Responsive: tablet ── */
        @media (max-width: 1024px) {
            .container { padding: 1.5rem 1.25rem 2.5rem; }
            .probability { font-size: 2.25rem; }
            .stats-bar { flex-wrap: wrap; }
            .stat-box { min-width: calc(33% - 0.5rem); }
            .detail-panel { width: 420px; right: -420px; }
        }

        @media (max-width: 700px) {
            html, body { overflow-x: hidden; }
            .search-form { flex-direction: column; }
            .stats-grid {
                grid-template-columns: repeat(3, 1fr);
                gap: 0.5rem;
            }
            .container { padding: 1.25rem 1rem 4.5rem; max-width: 100%; }
            .page-title { font-size: 1.35rem; line-height: 1.25; }
            .page-subtitle { font-size: 0.85rem; margin-bottom: 1.25rem; }

            /* Detail panel — full width, hide horizontal scroll, scale everything down */
            .detail-panel { width: 100%; right: -100%; overflow-x: hidden; }
            .detail-body { padding: 1.1rem 1rem 5rem; }
            .detail-body * { max-width: 100%; box-sizing: border-box; }
            .detail-body table { display: block; overflow-x: auto; max-width: 100%; }
            .detail-addr { font-size: 0.72rem; word-break: break-all; }
            .feature-bar-label { width: 110px; font-size: 0.68rem; }
            .feature-bar-val   { min-width: 44px; font-size: 0.68rem; }
            .tx-context { font-size: 0.72rem; word-break: break-all; }
            .list-panel { width: 100%; left: -100%; }

            /* Address feed — 3 col grid, shrinkable columns, clean ellipsis on address */
            .feed-header, .tx-row {
                grid-template-columns: minmax(0, 1fr) 60px 60px;
                font-size: 0.78rem;
                gap: 0.4rem;
                padding: 0.55rem 0.75rem;
            }
            .feed-header > *, .tx-row > * { min-width: 0; }
            .tx-row .address {
                font-size: 0.72rem;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                min-width: 0;
            }
            .feed-header > :last-child, .tx-row > :last-child { display: none; }

            .stream-header { flex-wrap: wrap; gap: 0.5rem; }
            .card { padding: 1.1rem; max-width: 100%; }

            /* Show the floating Gemini assistant */
            .mobile-chat-fab { display: flex !important; }

            /* Keep the inline "Ask AI Advisor" button visible inside the detail panel — user requested */
            .chat-toggle-btn { padding: 0.85rem; }
        }

        /* ── Responsive: small mobile ── */
        @media (max-width: 480px) {
            .container { padding: 0.85rem 0.65rem 4.5rem; }
            .page-title { font-size: 1.1rem; }
            .page-subtitle { font-size: 0.8rem; margin-bottom: 1rem; }
            .probability { font-size: 1.85rem; }
            .stats-grid {
                grid-template-columns: repeat(3, 1fr);
                gap: 0.4rem;
            }
            .stat-box { padding: 0.55rem 0.6rem; min-width: 0; }
            .stat-box .label { font-size: 0.62rem; }
            .stat-box .value { font-size: 0.95rem; }
            .uni-header {
                padding: 0.55rem 0.75rem;
                font-size: 0.72rem;
                gap: 0.5rem;
                flex-wrap: wrap;
            }
            .uni-header .divider { height: 0.95rem; }
            /* Toggle buttons in header collapse to icon-only on tiny screens */
            #modeToggle, #themeToggle {
                padding: 0.3rem 0.5rem !important;
                font-size: 0.7rem !important;
                gap: 0.2rem !important;
            }
            #modeLabel, #themeLabel { display: none; }
            .card { padding: 1rem; max-width: 100%; }
            .toast-container { left: 0.5rem; right: 0.5rem; max-width: none; bottom: 5rem; }
            .feed-header, .tx-row {
                grid-template-columns: minmax(0, 1fr) 55px 55px;
                font-size: 0.72rem;
                padding: 0.5rem 0.65rem;
                gap: 0.35rem;
            }
            .tx-row .address { font-size: 0.66rem; }
            .detail-body { padding: 1rem 0.75rem 5rem; }
            .feature-bar-label { width: 90px; }
        }

        /* ════════════════════════════════════════════════════════════
           Floating Gemini AI Assistant (mobile-only)
           ════════════════════════════════════════════════════════════ */
        .mobile-chat-fab {
            display: none;
            position: fixed;
            bottom: 1.1rem;
            right: 1.1rem;
            width: 58px;
            height: 58px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent), #7c1421);
            color: white;
            border: none;
            font-size: 1.5rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.35), 0 0 0 4px rgba(177,32,46,0.15);
            z-index: 180;
            cursor: pointer;
            align-items: center;
            justify-content: center;
            transition: transform 0.15s;
        }
        .mobile-chat-fab:active { transform: scale(0.92); }
        .mobile-chat-fab .fab-dot {
            position: absolute;
            top: 6px; right: 6px;
            width: 10px; height: 10px;
            background: #22c55e;
            border-radius: 50%;
            border: 2px solid var(--accent);
        }
        .mobile-chat-panel {
            display: none;
            position: fixed;
            inset: 0;
            z-index: 220;
            background: var(--bg);
            flex-direction: column;
        }
        .mobile-chat-panel.open { display: flex; }
        .mobile-chat-panel .mc-head {
            background: var(--accent);
            color: white;
            padding: 0.85rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-weight: 600;
            font-size: 1rem;
        }
        .mobile-chat-panel .mc-head .mc-title { flex: 1; }
        .mobile-chat-panel .mc-head button {
            background: rgba(255,255,255,0.18);
            border: none;
            color: white;
            width: 32px; height: 32px;
            border-radius: 50%;
            font-size: 1.1rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .mobile-chat-panel .mc-context {
            background: var(--card);
            border-bottom: 1px solid var(--border);
            padding: 0.55rem 1rem;
            font-size: 0.75rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .mobile-chat-panel .mc-body {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
        }

        /* Mobile chat: address picker (empty state) */
        .mc-addr-picker {
            padding: 1rem;
            background: var(--card);
            border-bottom: 1px solid var(--border);
            display: none;
            flex-direction: column;
            gap: 0.65rem;
        }
        .mc-addr-picker.show { display: flex; }
        .mc-addr-blurb {
            margin: 0;
            font-size: 0.85rem;
            color: var(--text-muted);
            line-height: 1.4;
        }
        .mc-addr-picker input {
            padding: 0.7rem 0.85rem;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            background: var(--bg);
            color: var(--text);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }
        .mc-addr-picker button {
            padding: 0.75rem;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 0.5rem;
            font-weight: 600;
            cursor: pointer;
            font-size: 0.95rem;
        }
        .mc-addr-picker button:disabled {
            opacity: 0.6; cursor: wait;
        }
        .mc-addr-hint {
            margin: 0;
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .mc-addr-hint a { color: var(--accent); text-decoration: none; }

        .mobile-chat-panel .mc-input {
            display: flex;
            gap: 0.5rem;
            padding: 0.75rem;
            border-top: 1px solid var(--border);
            background: var(--card);
        }
        .mobile-chat-panel .mc-input input {
            flex: 1;
            padding: 0.7rem 0.85rem;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            background: var(--bg);
            color: var(--text);
            font-size: 0.95rem;
            font-family: 'Inter', sans-serif;
        }
        .mobile-chat-panel .mc-input button {
            padding: 0.7rem 1.1rem;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 0.5rem;
            font-weight: 600;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="uni-header">
        <span>University of Malta</span>
        <span class="divider"></span>
        <span>Faculty of Engineering</span>
        <span class="divider"></span>
        <span>Sebastian Aquilina</span>
        <span class="divider"></span>
        <button id="modeToggle" onclick="toggleScoringMode()" style="margin-left:auto;background:none;border:1px solid rgba(255,255,255,0.3);border-radius:0.375rem;color:white;padding:0.35rem 0.75rem;cursor:pointer;font-size:0.85rem;display:flex;align-items:center;gap:0.4rem;transition:background 0.2s;" title="Switch between fixed and adaptive ensemble weighting">
            <span id="modeIcon">&#9878;</span> <span id="modeLabel">Fixed Weighting</span>
        </button>
        <button id="themeToggle" onclick="toggleTheme()" style="background:none;border:1px solid rgba(255,255,255,0.3);border-radius:0.375rem;color:white;padding:0.35rem 0.75rem;cursor:pointer;font-size:0.85rem;display:flex;align-items:center;gap:0.4rem;transition:background 0.2s;">
            <span id="themeIcon">&#9790;</span> <span id="themeLabel">Dark</span>
        </button>
    </div>

    <div class="container">
        <h1 class="page-title">AI Model for Fraud Detection in Blockchain Transactions</h1>
        <p class="page-subtitle">
            Live transaction analysis using a 3-model ensemble (XGBoost + Random Forest + Isolation Forest).
            Enter an Ethereum address to classify its fraud risk, or monitor the live blockchain stream below.
        </p>

        <!-- ── Manual search ── -->
        <div class="card">
            <form class="search-form" onsubmit="scoreAddress(event)">
                <div class="input-wrap">
                    <label for="address">Ethereum Address</label>
                    <input type="text" id="address" placeholder="0x..."
                           value="" autocomplete="off" spellcheck="false">
                </div>
                <div class="btn-wrap">
                    <button type="submit" id="submitBtn">Analyse</button>
                </div>
            </form>
        </div>

        <div id="loading" class="card loading" style="display:none;">
            <div class="spinner"></div>
            <p>Fetching transaction data from Etherscan&hellip;</p>
            <p style="font-size:0.8rem; margin-top:0.5rem; color:var(--text-muted);">This may take a few seconds</p>
        </div>

        <div id="result" class="card result">
            <div class="section-label">Fraud Probability</div>
            <div class="result-header">
                <div><div id="probability" class="probability">--</div></div>
                <div id="riskBadge" class="risk-badge">--</div>
            </div>
            <div class="risk-gauge">
                <div id="riskIndicator" class="risk-indicator" style="left:0%;"></div>
            </div>
            <div class="gauge-labels"><span>Low risk</span><span>High risk</span></div>
            <div style="margin-top:1.5rem;">
                <div class="section-label">Address Statistics</div>
                <div class="stats-grid">
                    <div class="stat-item"><div class="stat-label">Total Transactions</div><div id="statTxs" class="stat-value">--</div></div>
                    <div class="stat-item"><div class="stat-label">ERC-20 Transfers</div><div id="statErc20" class="stat-value">--</div></div>
                    <div class="stat-item"><div class="stat-label">ETH Sent</div><div id="statSent" class="stat-value">--</div></div>
                    <div class="stat-item"><div class="stat-label">ETH Received</div><div id="statReceived" class="stat-value">--</div></div>
                </div>
            </div>
            <div>
                <div class="section-label">Individual Model Scores</div>
                <div id="modelScores" class="model-scores-list"></div>
            </div>
        </div>

        <div id="error" class="card error" role="alert" style="display:none;"></div>

        <!-- ── Live stream ── -->
        <div class="stream-section">
            <div class="stream-header">
                <div style="display:flex;align-items:center;gap:0.75rem;">
                    <div class="stream-title">Live Blockchain Stream</div>
                    <button class="btn-secondary" onclick="exportCSV()" title="Export scored addresses to CSV">&#8681; Export</button>
                </div>
                <div style="display:flex;align-items:center;gap:0.5rem;">
                    <div class="status-badge">
                        <span class="status-dot disconnected" id="statusDot"></span>
                        <span id="statusText">Connecting&hellip;</span>
                    </div>
                    <button class="btn-secondary" onclick="clearSession()" title="Clear session data" style="font-size:0.7rem;padding:0.25rem 0.5rem;">Clear</button>
                </div>
            </div>
            <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:1rem; line-height:1.5;">
                A sample of high-value addresses is selected from each block and scored individually via the Etherscan API.
                Due to rate limits, only a subset of transactions per block is displayed.
            </p>

            <div class="stats-bar">
                <div class="stat-box" role="button" tabindex="0" onclick="openListPanel('summary',this)"><div class="label">Blocks</div><div class="value" id="sBlocks">0</div></div>
                <div class="stat-box" role="button" tabindex="0" onclick="openListPanel('summary',this)"><div class="label">Transactions</div><div class="value" id="sTxs">0</div></div>
                <div class="stat-box" role="button" tabindex="0" onclick="openListPanel('scored',this)"><div class="label">Scored</div><div class="value" id="sScored">0</div></div>
                <div class="stat-box" role="button" tabindex="0" onclick="openListPanel('medium',this)"><div class="label">Medium Risk</div><div class="value" id="sMedium" style="color:var(--warning);">0</div></div>
                <div class="stat-box" role="button" tabindex="0" onclick="openListPanel('alerts',this)"><div class="label">Alerts</div><div class="value" id="sAlerts" style="color:var(--danger);">0</div></div>
            </div>

            <div class="feed" id="feed" role="log" aria-live="polite" aria-label="Live transaction feed">
                <div class="feed-header">
                    <div>Address</div>
                    <div style="text-align:right;">Fraud %</div>
                    <div style="text-align:center;">Risk</div>
                    <div style="text-align:right;">Time</div>
                </div>
                <div class="empty-state" id="emptyState">
                    <div class="spinner"></div>
                    <p>Waiting for new blocks&hellip;</p>
                    <p style="margin-top:0.4rem; font-size:0.8rem;">
                        Ethereum produces a block every ~12 s.
                    </p>
                </div>
            </div>
        </div>

        <!-- ── Footer ── -->
        <div class="footer">
            <strong>Sebastian Aquilina</strong><br>
            B.Eng. (Hons.) in Electrical &amp; Electronic Engineering<br>
            University of Malta &mdash; 2026
            <hr>
            Supervised by <strong>Dr. Luana Chetcuti Zammit</strong>
        </div>
    </div>

    
    <!-- ════════════════ Mobile-only Gemini Assistant FAB ════════════════ -->
    <button class="mobile-chat-fab" id="mobileChatFab" onclick="openMobileChat()" aria-label="Open AI Assistant" title="Ask AI Advisor">
        &#129302;<span class="fab-dot"></span>
    </button>

    <div class="mobile-chat-panel" id="mobileChatPanel" role="dialog" aria-label="AI Advisor chat">
        <div class="mc-head">
            <span>&#129302;</span>
            <span class="mc-title">AI Advisor</span>
            <button onclick="closeMobileChat()" aria-label="Close">&times;</button>
        </div>
        <div class="mc-context" id="mcContext">No address analysed yet</div>

        <!-- Empty-state: address picker shown when no analysis exists -->
        <div class="mc-addr-picker" id="mcAddrPicker" style="display:none;">
            <p class="mc-addr-blurb">Paste an Ethereum address and I'll analyse it, then answer your questions about it.</p>
            <input id="mcAddrInput" type="text" placeholder="0x… (Ethereum address)" autocomplete="off"
                   onkeydown="if(event.key==='Enter')mcAnalyseAndChat()">
            <button onclick="mcAnalyseAndChat()" id="mcAnalyseBtn">Analyse &amp; chat</button>
            <p class="mc-addr-hint">Try Vitalik's wallet:
                <a href="#" onclick="document.getElementById('mcAddrInput').value='0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045';return false;">
                    0xd8dA…6045
                </a>
            </p>
        </div>

        <div class="mc-body" id="mobileChatMessages"></div>
        <div class="mc-input" id="mcInputBar">
            <input id="mobileChatInput" type="text" placeholder="Ask about this analysis…"
                   onkeydown="if(event.key==='Enter')sendMobileChat()">
            <button onclick="sendMobileChat()">Send</button>
        </div>
    </div>
    <!-- ── Toast notifications ── -->
    <div class="toast-container" id="toastContainer"></div>

    <!-- ── List slide-out panel (left) ── -->
    <div class="list-overlay" id="listOverlay"></div>
    <div class="list-panel" id="listPanel" role="dialog" aria-modal="true" aria-label="Address list panel">
        <div class="list-panel-header">
            <h3 id="listTitle">Addresses</h3>
            <button class="list-panel-close" id="listClose">&times;</button>
        </div>
        <div id="listBody"></div>
    </div>

    <!-- ── Slide-out detail panel ── -->
    <div class="detail-overlay" id="detailOverlay"></div>
    <div class="detail-panel" id="detailPanel" role="dialog" aria-modal="true" aria-label="Address detail panel">
        <button class="detail-close" id="detailClose">&times;</button>
        <div class="detail-body" id="detailBody">
            <!-- filled dynamically -->
        </div>
    </div>

    <script>
    /* ================================================================
       Theme toggle
       ================================================================ */
    (function() {
        const saved = localStorage.getItem('ethfd_theme');
        if (saved) document.documentElement.dataset.theme = saved;
        else if (window.matchMedia('(prefers-color-scheme: dark)').matches)
            document.documentElement.dataset.theme = 'dark';
        updateThemeBtn();
    })();
    function toggleTheme() {
        const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
        document.documentElement.dataset.theme = next;
        localStorage.setItem('ethfd_theme', next);
        updateThemeBtn();
    }
    function updateThemeBtn() {
        const isDark = document.documentElement.dataset.theme === 'dark';
        const icon = document.getElementById('themeIcon');
        const label = document.getElementById('themeLabel');
        if (icon) icon.innerHTML = isDark ? '&#9788;' : '&#9790;';
        if (label) label.textContent = isDark ? 'Light' : 'Dark';
    }

    /* ================================================================
       Scoring mode toggle
       ================================================================ */
    let currentScoringMode = 'fixed';

    async function fetchScoringMode() {
        try {
            const r = await fetch('/api/scoring-mode');
            const d = await r.json();
            currentScoringMode = d.mode;
            updateModeBtn();
        } catch(e) {}
    }
    fetchScoringMode();

    async function toggleScoringMode() {
        const next = currentScoringMode === 'fixed' ? 'adaptive' : 'fixed';
        try {
            const r = await fetch('/api/scoring-mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: next})
            });
            const d = await r.json();
            if (d.mode) {
                currentScoringMode = d.mode;
                updateModeBtn();
            }
        } catch(e) { console.error('Mode switch failed:', e); }
    }

    function updateModeBtn() {
        const btn = document.getElementById('modeToggle');
        const icon = document.getElementById('modeIcon');
        const label = document.getElementById('modeLabel');
        if (!btn) return;
        if (currentScoringMode === 'adaptive') {
            icon.innerHTML = '&#9881;';
            label.textContent = 'Adaptive Weighting';
            btn.classList.add('active-adaptive');
        } else {
            icon.innerHTML = '&#9878;';
            label.textContent = 'Fixed Weighting';
            btn.classList.remove('active-adaptive');
        }
    }

    function renderAdaptiveWeights(weights) {
        if (!weights) return '';
        return '<div class="adaptive-weights">' +
            '<div class="aw-item"><div class="aw-label">XGBoost</div><div class="aw-val">' + (weights.w_xgb * 100).toFixed(1) + '%</div></div>' +
            '<div class="aw-item"><div class="aw-label">Random Forest</div><div class="aw-val">' + (weights.w_rf * 100).toFixed(1) + '%</div></div>' +
            '<div class="aw-item" style="border-left:2px solid var(--border);"><div class="aw-label">Isolation Forest</div><div class="aw-val" style="color:#10b981;">' + (weights.w_if * 100).toFixed(1) + '%</div></div>' +
            '</div>';
    }

    function renderModeIndicator(mode) {
        var cls = mode === 'adaptive' ? 'adaptive' : 'fixed';
        var label = mode === 'adaptive' ? 'Adaptive' : 'Fixed';
        return '<span class="mode-indicator ' + cls + '">' + label + '</span>';
    }

    async function compareAddress(addr) {
        var container = document.getElementById('compareResult');
        if (!container) return;
        container.innerHTML = '<div style="text-align:center;padding:0.75rem;color:var(--text-muted);font-size:0.8rem;">Comparing modes\u2026</div>';
        try {
            var r = await fetch('/api/compare/' + encodeURIComponent(addr));
            var d = await r.json();
            if (d.error) { container.innerHTML = '<div style="color:var(--danger);">Error: ' + d.error + '</div>'; return; }
            var f = d.fixed, a = d.adaptive;
            var fms = f.model_scores || {}, ams = a.model_scores || {};
            var aw = a.adaptive_weights || {};
            var html = '<table>';
            html += '<tr><th></th><th>Fixed</th><th>Adaptive</th><th>\u0394</th></tr>';
            html += '<tr><td>Ensemble Score</td><td>' + (f.fraud_probability*100).toFixed(2) + '%</td><td>' + (a.fraud_probability*100).toFixed(2) + '%</td><td style="color:' + (a.fraud_probability > f.fraud_probability ? 'var(--danger)' : 'var(--success)') + '">' + ((a.fraud_probability - f.fraud_probability)*100).toFixed(2) + '%</td></tr>';
            html += '<tr><td>Risk Level</td><td><span class="risk-pill risk-'+f.risk_level+'">'+f.risk_level+'</span></td><td><span class="risk-pill risk-'+a.risk_level+'">'+a.risk_level+'</span></td><td></td></tr>';
            html += '<tr><td>IF Weight</td><td>5.0%</td><td>' + (aw.w_if ? (aw.w_if*100).toFixed(1)+'%' : 'N/A') + '</td><td></td></tr>';
            html += '</table>';
            container.innerHTML = html;
        } catch(e) { container.innerHTML = '<div style="color:var(--danger);">Network error</div>'; }
    }

    /* ================================================================
       Utility functions
       ================================================================ */
    function animateCounter(element, target, duration) {
        duration = duration || 800;
        const start = performance.now();
        function step(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            element.textContent = (target * eased).toFixed(1) + '%';
            if (progress < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }

    function relativeTime(ts) {
        const diff = Math.floor((Date.now() - ts) / 1000);
        if (diff < 5) return 'just now';
        if (diff < 60) return diff + 's ago';
        if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
        if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
        return Math.floor(diff / 86400) + 'd ago';
    }
    function updateRelativeTimes() {
        document.querySelectorAll('.time-cell[data-ts]').forEach(function(el) {
            el.textContent = relativeTime(parseInt(el.dataset.ts));
        });
    }
    setInterval(updateRelativeTimes, 10000);

    function getAgreementHtml(modelScores) {
        var vals = Object.values(modelScores).filter(function(v) { return v !== null; });
        if (vals.length < 2) return '';
        var mean = vals.reduce(function(a, b) { return a + b; }, 0) / vals.length;
        var stdev = Math.sqrt(vals.reduce(function(s, v) { return s + (v - mean) * (v - mean); }, 0) / vals.length);
        var cls, label;
        if (stdev < 0.1) { cls = 'strong'; label = 'Strong model agreement'; }
        else if (stdev < 0.2) { cls = 'partial'; label = 'Partial model agreement'; }
        else { cls = 'weak'; label = 'Model disagreement \u2014 interpret with caution'; }
        return '<div class="model-agreement ' + cls + '"><span class="agreement-dot"></span> ' + label + '</div>';
    }

    function showToast(addr, prob, risk) {
        var container = document.getElementById('toastContainer');
        var toast = document.createElement('div');
        toast.className = 'toast' + (risk === 'MEDIUM' ? ' medium' : '');
        var shortAddr = addr.substring(0, 8) + '...' + addr.substring(addr.length - 6);
        var color = prob >= 0.7 ? 'var(--danger)' : 'var(--warning)';
        toast.innerHTML =
            '<div class="toast-icon">' + (prob >= 0.9 ? '&#9888;' : '&#128270;') + '</div>' +
            '<div class="toast-body"><div class="toast-addr">' + shortAddr + '</div>' +
            '<div class="toast-prob" style="color:' + color + '">' + (prob * 100).toFixed(1) + '% \u2014 ' + risk + '</div></div>' +
            '<button class="toast-dismiss" onclick="this.parentElement.remove()">&#10005;</button>';
        toast.onclick = function(e) { if (!e.target.classList.contains('toast-dismiss')) { openDetail(addr); toast.remove(); } };
        container.appendChild(toast);
        setTimeout(function() { if (toast.parentNode) { toast.style.animation = 'toastOut 0.3s ease-out forwards'; setTimeout(function() { toast.remove(); }, 300); } }, 6000);
        if (prob >= 0.9) playAlertBeep();
        while (container.children.length > 5) container.removeChild(container.firstChild);
    }

    function playAlertBeep() {
        try {
            var ctx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.connect(gain); gain.connect(ctx.destination);
            osc.frequency.value = 520; osc.type = 'sine';
            gain.gain.value = 0.08;
            osc.start(); osc.stop(ctx.currentTime + 0.15);
        } catch(e) {}
    }

    function copyAddr(addr, btn) {
        navigator.clipboard.writeText(addr).then(function() {
            btn.classList.add('copied');
            btn.innerHTML = '&#10003;';
            setTimeout(function() { btn.classList.remove('copied'); btn.innerHTML = '&#128203;'; }, 1500);
        });
    }

    function exportCSV() {
        var entries = Object.values(dataStore);
        if (!entries.length) { alert('No scored addresses to export.'); return; }
        var headers = ['Address','Fraud Probability','Risk Level','Ensemble Method','XGBoost','Random Forest','Isolation Forest','Total Transactions','ERC-20 Transfers','ETH Sent','ETH Received','Balance ETH','Scored At'];
        var rows = entries.map(function(d) {
            var ms = d.model_scores || {};
            var as = d.address_stats || {};
            var ctx = d.stream_context || {};
            return [
                d.address,
                (d.fraud_probability || 0).toFixed(4),
                d.risk_level || '',
                d.ensemble_method || '',
                ms.xgboost != null ? ms.xgboost.toFixed(4) : '',
                ms.random_forest != null ? ms.random_forest.toFixed(4) : '',
                ms.isolation_forest != null ? ms.isolation_forest.toFixed(4) : '',
                as.total_transactions != null ? as.total_transactions : '',
                as.total_erc20_transfers != null ? as.total_erc20_transfers : '',
                as.total_ether_sent != null ? as.total_ether_sent.toFixed(4) : '',
                as.total_ether_received != null ? as.total_ether_received.toFixed(4) : '',
                as.balance_eth != null ? as.balance_eth.toFixed(4) : '',
                ctx.scored_at || ''
            ].map(function(v) { return '"' + String(v).replace(/"/g, '""') + '"'; }).join(',');
        });
        var csv = headers.join(',') + '\\n' + rows.join('\\n');
        var blob = new Blob([csv], { type: 'text/csv' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url; a.download = 'fraud_scores_' + new Date().toISOString().slice(0,10) + '.csv';
        a.click(); URL.revokeObjectURL(url);
    }

    /* ================================================================
       Manual address scoring
       ================================================================ */
    async function scoreAddress(e) {
        e.preventDefault();
        const address = document.getElementById('address').value.trim();
        if (!address) return;
        const loading = document.getElementById('loading');
        const result  = document.getElementById('result');
        const error   = document.getElementById('error');
        const btn     = document.getElementById('submitBtn');

        result.classList.remove('visible');
        error.style.display = 'none';
        loading.innerHTML = '<div class="skeleton skeleton-line w-40" style="margin-bottom:0.75rem;"></div>' +
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1.25rem;">' +
            '<div class="skeleton skeleton-block big" style="width:180px;"></div>' +
            '<div class="skeleton" style="width:80px;height:32px;border-radius:2rem;"></div></div>' +
            '<div class="skeleton" style="height:10px;border-radius:5px;margin-bottom:1.5rem;"></div>' +
            '<div class="skeleton skeleton-line w-40" style="margin-bottom:0.75rem;"></div>' +
            '<div class="skeleton-grid">' +
            '<div class="skeleton skeleton-grid-item"></div><div class="skeleton skeleton-grid-item"></div>' +
            '<div class="skeleton skeleton-grid-item"></div><div class="skeleton skeleton-grid-item"></div></div>' +
            '<div class="skeleton skeleton-line w-40" style="margin-bottom:0.75rem;"></div>' +
            '<div class="skeleton skeleton-line w-80"></div><div class="skeleton skeleton-line w-60"></div>';
        loading.style.display = 'block';
        btn.disabled = true;

        try {
            const resp = await fetch('/api/score/' + encodeURIComponent(address));
            const data = await resp.json();
            loading.style.display = 'none';
            if (data.error) { error.textContent = data.error; error.style.display = 'block'; return; }
            fillManualResult(data);
            result.classList.add('visible');
        } catch (err) {
            loading.style.display = 'none';
            error.textContent = 'Network error: ' + err.message;
            error.style.display = 'block';
        } finally { btn.disabled = false; }
    }

    function fillManualResult(d) {
        const p = d.fraud_probability;
        const probEl = document.getElementById('probability');
        probEl.textContent = '0.0%';
        animateCounter(probEl, p * 100);
        const b = document.getElementById('riskBadge');
        b.textContent = d.risk_level; b.className = 'risk-badge risk-' + d.risk_level;
        document.getElementById('riskIndicator').style.left = (p*100) + '%';
        document.getElementById('statTxs').textContent    = d.address_stats.total_transactions;
        document.getElementById('statErc20').textContent   = d.address_stats.total_erc20_transfers;
        document.getElementById('statSent').textContent    = d.address_stats.total_ether_sent.toFixed(4) + ' ETH';
        document.getElementById('statReceived').textContent = d.address_stats.total_ether_received.toFixed(4) + ' ETH';
        const names = { xgboost: 'XGBoost (calibrated)', random_forest: 'Random Forest (calibrated)', isolation_forest: 'Isolation Forest (unsupervised)' };
        const supervisedKeys = ['xgboost', 'random_forest'];
        const unsupervisedKeys = ['isolation_forest'];
        let h = '<div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.35rem;">Supervised Models</div>';
        for (const k of supervisedKeys)
            if (d.model_scores[k] !== null && d.model_scores[k] !== undefined) { const s = d.model_scores[k]; h += '<div><span>'+(names[k])+'</span><div class="score-bar-wrap"><div class="score-bar-bg"><div class="score-bar-fill" style="width:'+(s*100)+'%;background:'+probColor(s)+'"></div></div></div><span class="score-val" style="color:'+probColor(s)+'">'+(s*100).toFixed(2)+'%</span></div>'; }
        h += '<div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;margin:0.5rem 0 0.35rem;border-top:1px solid var(--border);padding-top:0.5rem;">Unsupervised Model</div>';
        for (const k of unsupervisedKeys)
            if (d.model_scores[k] !== null && d.model_scores[k] !== undefined) { const s = d.model_scores[k]; h += '<div><span>'+(names[k])+'</span><div class="score-bar-wrap"><div class="score-bar-bg"><div class="score-bar-fill" style="width:'+(s*100)+'%;background:'+probColor(s)+'"></div></div></div><span class="score-val" style="color:'+probColor(s)+'">'+(s*100).toFixed(2)+'%</span></div>'; }
        h += getAgreementHtml(d.model_scores);
        var em = d.ensemble_method || '';
        var emLabel = em === 'weighted_average_3model' ? '3-Model Weighted Ensemble' : em === 'weighted_average_2model' ? '2-Model Weighted Ensemble' : em === 'xgboost_only' ? 'XGBoost Only' : em;
        h += '<div style="margin-top:0.6rem;font-size:0.78rem;color:var(--text-muted);padding:0.5rem 0.75rem;background:var(--bg);border-radius:0.35rem;"><span style="font-weight:600;">Ensemble method:</span> ' + emLabel + '</div>';
        if (em === 'weighted_average_3model') h += '<div style="font-size:0.7rem;color:var(--text-muted);padding:0.25rem 0.75rem;font-family:JetBrains Mono,monospace;" title="Ensemble formula">0.95 &times; (0.190&middot;XGB + 0.810&middot;RF) + 0.05 &times; IF</div>';
        var mode = d.scoring_mode || 'fixed';
        h += '<div style="margin-top:0.5rem;display:flex;align-items:center;gap:0.5rem;">' + renderModeIndicator(mode) + '</div>';
        if (d.adaptive_weights) {
            h += '<div style="margin-top:0.25rem;"><div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.25rem;">Per-Address Weights</div>' + renderAdaptiveWeights(d.adaptive_weights) + '</div>';
        }
        document.getElementById('modelScores').innerHTML = h;
    }

    /* ================================================================
       Live stream
       ================================================================ */
    const feed      = document.getElementById('feed');
    const empty     = document.getElementById('emptyState');
    const sDot      = document.getElementById('statusDot');
    const sText     = document.getElementById('statusText');
    let totScored = 0, totAlerts = 0, totMedium = 0;
    const dataStore = {};  // addr -> full result payload

    function probColor(p) {
        return p >= 0.7 ? 'var(--danger)' : p >= 0.5 ? 'var(--warning)' : 'var(--success)';
    }

    function connectSSE() {
        const es = new EventSource('/api/stream');
        es.onopen = () => { sDot.className='status-dot connected'; sText.textContent='Connected \u2014 streaming live'; };
        es.onerror = () => { sDot.className='status-dot disconnected'; sText.textContent='Reconnecting\u2026'; };

        /* Scored result — each row arrives fully scored */
        es.addEventListener('score', (ev) => {
            const d = JSON.parse(ev.data);
            if (d.error) return;
            if (empty && empty.parentNode) empty.remove();

            totScored++;
            const prob = d.fraud_probability || 0;
            const isAlert = prob >= 0.7;
            const isMedium = prob >= 0.5 && prob < 0.7;
            if (isAlert) totAlerts++;
            if (isMedium) totMedium++;
            document.getElementById('sScored').textContent = totScored;
            document.getElementById('sAlerts').textContent  = totAlerts;
            document.getElementById('sMedium').textContent  = totMedium;

            const addr = d.address || '?';
            dataStore[addr] = d;

            const p    = d.fraud_probability || 0;
            const risk = d.risk_level || 'UNKNOWN';

            if (isAlert) showToast(addr, p, risk);

            const rowClass = isAlert ? ' alert' : isMedium ? ' medium' : '';
            const row = document.createElement('div');
            row.className = 'tx-row new' + rowClass;
            row.dataset.addr = addr;
            row.setAttribute('tabindex', '0');
            row.setAttribute('role', 'row');
            row.onclick = () => openDetail(addr);
            row.innerHTML =
                '<div class="address" title="'+addr+'">'+addr+'</div>' +
                '<div class="prob" style="text-align:right;color:'+probColor(p)+'">'+(p*100).toFixed(1)+'%</div>' +
                '<div style="text-align:center"><span class="risk-pill risk-'+risk+'">'+risk+'</span></div>' +
                '<div class="time-cell" data-ts="'+Date.now()+'">just now</div>';

            const header = feed.querySelector('.feed-header');
            if (header && header.nextSibling) feed.insertBefore(row, header.nextSibling);
            else feed.appendChild(row);

            while (feed.children.length > 201) feed.removeChild(feed.lastChild);

            // Live-update list panel if open
            if (activeListCategory) refreshListPanel();
            saveSession();
        });
    }

    async function pollStats() {
        try {
            const r = await fetch('/api/stream/stats');
            if (r.ok) {
                const s = await r.json();
                document.getElementById('sBlocks').textContent = s.blocks_processed || 0;
                document.getElementById('sTxs').textContent    = s.transactions_seen || 0;
            }
        } catch(e) {}
    }

    /* ================================================================
       List panel (left side)
       ================================================================ */
    const listOverlay = document.getElementById('listOverlay');
    const listPanel   = document.getElementById('listPanel');
    const listTitle   = document.getElementById('listTitle');
    const listBody    = document.getElementById('listBody');
    let activeListCategory = null;

    function closeListPanel() {
        listOverlay.classList.remove('open');
        listPanel.classList.remove('open');
        document.querySelectorAll('.stat-box').forEach(b => b.classList.remove('active'));
        activeListCategory = null;
    }
    document.getElementById('listClose').onclick = closeListPanel;
    listOverlay.onclick = closeListPanel;

    function openListPanel(category, el) {
        if (activeListCategory === category) { closeListPanel(); return; }
        activeListCategory = category;
        document.querySelectorAll('.stat-box').forEach(b => b.classList.remove('active'));
        if (el) el.classList.add('active');
        const titles = { summary:'Stream Summary', scored:'All Scored Addresses', medium:'Medium Risk Addresses', alerts:'High-Risk Alerts' };
        listTitle.textContent = titles[category] || 'Addresses';
        refreshListPanel();
        listOverlay.classList.add('open');
        listPanel.classList.add('open');
        document.getElementById('listClose').focus();
    }

    function refreshListPanel() {
        const cat = activeListCategory;
        if (!cat) return;
        if (cat === 'summary') {
            listBody.innerHTML = '<div style="padding:1.5rem;line-height:2;"><div class="section-label">Stream Statistics</div>' +
                '<div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid var(--border)"><span>Blocks processed</span><strong>'+document.getElementById('sBlocks').textContent+'</strong></div>' +
                '<div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid var(--border)"><span>Total transactions</span><strong>'+document.getElementById('sTxs').textContent+'</strong></div>' +
                '<div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid var(--border)"><span>Addresses scored</span><strong>'+totScored+'</strong></div>' +
                '<div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid var(--border)"><span>Medium risk</span><strong style="color:var(--warning)">'+totMedium+'</strong></div>' +
                '<div style="display:flex;justify-content:space-between;padding:0.5rem 0"><span>High-risk alerts</span><strong style="color:var(--danger)">'+totAlerts+'</strong></div></div>';
            return;
        }
        const entries = Object.values(dataStore).filter(d => {
            const p = d.fraud_probability || 0;
            if (cat === 'alerts') return p >= 0.7;
            if (cat === 'medium') return p >= 0.5 && p < 0.7;
            return true;
        }).sort((a, b) => (b.fraud_probability||0) - (a.fraud_probability||0));
        if (!entries.length) { listBody.innerHTML = '<div class="list-empty">No addresses in this category yet.</div>'; return; }
        let html = '';
        for (const d of entries) {
            const addr = d.address||'?', p = d.fraud_probability||0, risk = d.risk_level||'UNKNOWN';
            html += '<div class="list-item risk-c-'+risk+'" onclick="openDetail(`'+addr+'`)"><div class="li-addr" title="'+addr+'">'+addr+'</div><div class="li-prob" style="color:'+probColor(p)+'">'+(p*100).toFixed(1)+'%</div><div style="text-align:center"><span class="risk-pill risk-'+risk+'">'+risk+'</span></div></div>';
        }
        listBody.innerHTML = html;
    }

    /* Session persistence */
    function saveSession() {
        try {
            var session = { dataStore: dataStore, totScored: totScored, totAlerts: totAlerts, totMedium: totMedium };
            localStorage.setItem('ethfd_session', JSON.stringify(session));
        } catch(e) {}
    }
    function loadSession() {
        try {
            var raw = localStorage.getItem('ethfd_session');
            if (!raw) return;
            var session = JSON.parse(raw);
            if (session.dataStore) {
                Object.assign(dataStore, session.dataStore);
                totScored = session.totScored || 0;
                totAlerts = session.totAlerts || 0;
                totMedium = session.totMedium || 0;
                document.getElementById('sScored').textContent = totScored;
                document.getElementById('sAlerts').textContent = totAlerts;
                document.getElementById('sMedium').textContent = totMedium;
                var sorted = Object.values(dataStore).sort(function(a, b) {
                    return new Date(b.stream_context && b.stream_context.scored_at || 0) - new Date(a.stream_context && a.stream_context.scored_at || 0);
                }).slice(0, 50);
                if (sorted.length > 0 && empty && empty.parentNode) empty.remove();
                sorted.forEach(function(d) {
                    var addr = d.address || '?';
                    var p = d.fraud_probability || 0;
                    var risk = d.risk_level || 'UNKNOWN';
                    var isA = p >= 0.7, isM = p >= 0.5 && p < 0.7;
                    var rc = isA ? ' alert' : isM ? ' medium' : '';
                    var row = document.createElement('div');
                    row.className = 'tx-row' + rc;
                    row.dataset.addr = addr;
                    row.setAttribute('tabindex', '0');
                    row.setAttribute('role', 'row');
                    row.onclick = function() { openDetail(`${addr}`); };
                    var ts = d.stream_context && d.stream_context.scored_at ? new Date(d.stream_context.scored_at).toLocaleTimeString() : '';
                    row.innerHTML =
                        '<div class="address" title="'+addr+'">'+addr+'</div>' +
                        '<div class="prob" style="text-align:right;color:'+probColor(p)+'">'+(p*100).toFixed(1)+'%</div>' +
                        '<div style="text-align:center"><span class="risk-pill risk-'+risk+'">'+risk+'</span></div>' +
                        '<div class="time-cell">'+ts+'</div>';
                    var header = feed.querySelector('.feed-header');
                    feed.appendChild(row);
                });
            }
        } catch(e) { console.warn('Session load failed:', e); }
    }
    function clearSession() {
        localStorage.removeItem('ethfd_session');
        location.reload();
    }

    connectSSE();
    setInterval(pollStats, 5000);
    pollStats();
    loadSession();

    /* ================================================================
       Detail panel
       ================================================================ */
    const overlay = document.getElementById('detailOverlay');
    const panel   = document.getElementById('detailPanel');
    const body    = document.getElementById('detailBody');

    function closeDetail() {
        overlay.classList.remove('open');
        panel.classList.remove('open');
        document.querySelectorAll('.tx-row.selected').forEach(r => r.classList.remove('selected'));
    }
    document.getElementById('detailClose').onclick = closeDetail;
    overlay.onclick = closeDetail;
    let selectedFeedRow = -1;
    document.addEventListener('keydown', function(e) {
        var tag = document.activeElement.tagName;
        var isInput = tag === 'INPUT' || tag === 'TEXTAREA';

        if (e.key === 'Escape') {
            if (document.getElementById('shortcutOverlay').classList.contains('open')) {
                document.getElementById('shortcutOverlay').classList.remove('open');
            } else {
                closeDetail(); closeListPanel();
            }
            document.activeElement.blur();
            return;
        }

        if (isInput) return;

        if (e.key === '/') {
            e.preventDefault();
            document.getElementById('address').focus();
            return;
        }
        if (e.key === '?') {
            e.preventDefault();
            document.getElementById('shortcutOverlay').classList.toggle('open');
            return;
        }
        if (e.key === 'd' && !e.metaKey && !e.ctrlKey) {
            toggleTheme();
            return;
        }

        var rows = Array.from(feed.querySelectorAll('.tx-row'));
        if (!rows.length) return;

        if (e.key === 'j') {
            e.preventDefault();
            selectedFeedRow = Math.min(selectedFeedRow + 1, rows.length - 1);
            rows.forEach(function(r) { r.classList.remove('selected'); });
            rows[selectedFeedRow].classList.add('selected');
            rows[selectedFeedRow].scrollIntoView({ block: 'nearest' });
            return;
        }
        if (e.key === 'k') {
            e.preventDefault();
            selectedFeedRow = Math.max(selectedFeedRow - 1, 0);
            rows.forEach(function(r) { r.classList.remove('selected'); });
            rows[selectedFeedRow].classList.add('selected');
            rows[selectedFeedRow].scrollIntoView({ block: 'nearest' });
            return;
        }
        if (e.key === 'Enter' && selectedFeedRow >= 0 && selectedFeedRow < rows.length) {
            var a = rows[selectedFeedRow].dataset.addr;
            if (a) openDetail(a);
            return;
        }
        if (e.key === 'c' && selectedFeedRow >= 0 && selectedFeedRow < rows.length) {
            var ad = rows[selectedFeedRow].dataset.addr;
            if (ad) navigator.clipboard.writeText(ad);
            return;
        }
    });

    function openDetail(addr) {
        const d = dataStore[addr];
        if (!d) return;

        // Highlight row
        document.querySelectorAll('.tx-row.selected').forEach(r => r.classList.remove('selected'));
        const row = feed.querySelector('.tx-row[data-addr="'+CSS.escape(addr)+'"]');
        if (row) row.classList.add('selected');

        const p    = d.fraud_probability || 0;
        const risk = d.risk_level || 'UNKNOWN';
        const ms   = d.model_scores || {};
        const as   = d.address_stats || {};
        const ctx  = d.stream_context || {};
        const tx   = ctx.triggering_tx || {};

        let html = '';

        // Address with copy button
        html += '<div class="detail-addr"><a href="https://etherscan.io/address/'+addr+'" target="_blank" rel="noopener">'+addr+'</a>';
        html += '<button class="copy-btn" onclick="event.stopPropagation();copyAddr(`'+addr+'`,this)" title="Copy address">&#128203;</button></div>';

        // Fraud probability + gauge
        html += '<div class="detail-section">';
        html += '<div class="section-label">Fraud Probability</div>';
        html += '<div class="result-header">';
        html += '  <div><div class="probability" style="color:'+probColor(p)+'">0.0%</div></div>';
        html += '  <div><span class="risk-badge risk-'+risk+'">'+risk+'</span></div>';
        html += '</div>';
        html += '<div class="risk-gauge"><div class="risk-indicator" style="left:'+(p*100)+'%"></div></div>';
        html += '<div class="gauge-labels"><span>Low risk</span><span>High risk</span></div>';
        html += '</div>';

        // Model scores with sparkline bars
        html += '<div class="detail-section">';
        html += '<div class="section-label">Individual Model Scores</div>';
        html += '<div class="model-scores-list">';
        const dNames = { xgboost: 'XGBoost (calibrated)', random_forest: 'Random Forest (calibrated)', isolation_forest: 'Isolation Forest (unsupervised)' };
        const dSupervisedKeys = ['xgboost', 'random_forest'];
        const dUnsupervisedKeys = ['isolation_forest'];
        html += '<div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.35rem;">Supervised Models</div>';
        for (const k of dSupervisedKeys) {
            const v = ms[k];
            if (v !== null && v !== undefined) html += '<div><span>'+dNames[k]+'</span><div class="score-bar-wrap"><div class="score-bar-bg"><div class="score-bar-fill" style="width:'+(v*100)+'%;background:'+probColor(v)+'"></div></div></div><span class="score-val" style="color:'+probColor(v)+'">'+(v*100).toFixed(2)+'%</span></div>';
        }
        html += '<div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;margin:0.5rem 0 0.35rem;border-top:1px solid var(--border);padding-top:0.5rem;">Unsupervised Model</div>';
        for (const k of dUnsupervisedKeys) {
            const v = ms[k];
            if (v !== null && v !== undefined) html += '<div><span>'+dNames[k]+'</span><div class="score-bar-wrap"><div class="score-bar-bg"><div class="score-bar-fill" style="width:'+(v*100)+'%;background:'+probColor(v)+'"></div></div></div><span class="score-val" style="color:'+probColor(v)+'">'+(v*100).toFixed(2)+'%</span></div>';
        }
        html += '</div>';
        html += getAgreementHtml(ms);
        var dEm = d.ensemble_method || '';
        var dEmLabel = dEm === 'weighted_average_3model' ? '3-Model Weighted Ensemble' : dEm === 'weighted_average_2model' ? '2-Model Weighted Ensemble' : dEm === 'xgboost_only' ? 'XGBoost Only' : dEm;
        html += '<div style="margin-top:0.6rem;font-size:0.78rem;color:var(--text-muted);padding:0.5rem 0.75rem;background:var(--bg);border-radius:0.35rem;"><span style="font-weight:600;">Ensemble method:</span> ' + dEmLabel + '</div>';
        if (dEm === 'weighted_average_3model') html += '<div style="font-size:0.7rem;color:var(--text-muted);padding:0.25rem 0.75rem;font-family:JetBrains Mono,monospace;" title="Ensemble formula">0.95 &times; (0.190&middot;XGB + 0.810&middot;RF) + 0.05 &times; IF</div>';
        var dMode = d.scoring_mode || 'fixed';
        html += '<div style="margin-top:0.5rem;display:flex;align-items:center;gap:0.5rem;">' + renderModeIndicator(dMode);
        html += ' <button class="compare-btn" onclick="compareAddress(`'+addr+'`)">Compare Modes</button></div>';
        if (d.adaptive_weights) {
            html += '<div style="margin-top:0.25rem;"><div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.25rem;">Per-Address Weights</div>' + renderAdaptiveWeights(d.adaptive_weights) + '</div>';
        }
        html += '<div class="compare-result" id="compareResult"></div>';
        html += '</div>';

        // Address statistics
        html += '<div class="detail-section">';
        html += '<div class="section-label">Address Statistics</div>';
        html += '<div class="stats-grid">';
        html += '<div class="stat-item"><div class="stat-label">Total Transactions</div><div class="stat-value">'+(as.total_transactions || '--')+'</div></div>';
        html += '<div class="stat-item"><div class="stat-label">ERC-20 Transfers</div><div class="stat-value">'+(as.total_erc20_transfers || '--')+'</div></div>';
        html += '<div class="stat-item"><div class="stat-label">ETH Sent</div><div class="stat-value">'+(as.total_ether_sent != null ? as.total_ether_sent.toFixed(4)+' ETH' : '--')+'</div></div>';
        html += '<div class="stat-item"><div class="stat-label">ETH Received</div><div class="stat-value">'+(as.total_ether_received != null ? as.total_ether_received.toFixed(4)+' ETH' : '--')+'</div></div>';
        html += '<div class="stat-item"><div class="stat-label">Balance</div><div class="stat-value">'+(as.balance_eth != null ? as.balance_eth.toFixed(4)+' ETH' : '--')+'</div></div>';
        html += '</div></div>';

        // Feature importance
        html += '<div class="detail-section" id="featureSection">';
        html += '<div class="section-label">Top Contributing Features</div>';
        html += '<div id="featureBars" class="feature-bars"><div style="text-align:center;padding:1rem;color:var(--text-muted);font-size:0.8rem;">Loading features\u2026</div></div>';
        html += '</div>';

        // Triggering transaction context
        if (tx.hash) {
            html += '<div class="detail-section">';
            html += '<div class="section-label">Triggering Transaction</div>';
            html += '<div class="tx-context">';
            html += '<div><span class="ctx-label">Hash:</span> <a href="https://etherscan.io/tx/0x'+tx.hash+'" target="_blank" rel="noopener">0x'+tx.hash.substring(0,16)+'&hellip;</a></div>';
            html += '<div><span class="ctx-label">Block:</span> '+(tx.block || '--')+'</div>';
            html += '<div><span class="ctx-label">Value:</span> '+(tx.value_eth != null ? tx.value_eth+' ETH' : '--')+'</div>';
            html += '<div><span class="ctx-label">Time:</span> '+(tx.timestamp ? new Date(tx.timestamp).toLocaleString() : '--')+'</div>';
            html += '</div></div>';
        }

        // Scored at
        if (ctx.scored_at) {
            html += '<div style="margin-top:1rem; font-size:0.75rem; color:var(--text-muted);">Scored at: '+new Date(ctx.scored_at).toLocaleString()+'</div>';
        }

        // AI Advisor button + embedded chat
        html += '<button class="chat-toggle-btn" onclick="toggleChat(`' + addr + '`)">&#129302; Ask AI Advisor</button>';
        html += '<div class="chat-section" id="chatSection">';
        html += '<div class="section-label">AI Fraud Advisor</div>';
        html += '<div class="chat-messages" id="chatMessages">';
        html += '<div class="chat-bubble ai">Hello! I can help you understand why this address was flagged and what the fraud indicators mean. What would you like to know?</div>';
        html += '</div>';
        html += '<div class="chat-input-bar">';
        html += '<input type="text" id="chatInput" placeholder="Ask about this analysis\u2026" onkeydown="if(event.key===&#39;Enter&#39;)sendChat()">';
        html += '<button onclick="sendChat()">Send</button>';
        html += '</div>';
        html += '</div>';

        body.innerHTML = html;

        // Animate probability counter
        var probEl = body.querySelector('.probability');
        if (probEl) animateCounter(probEl, p * 100);

        // Load feature importance
        fetch('/api/features/' + encodeURIComponent(addr))
            .then(function(r) { return r.json(); })
            .then(function(data) {
                var container = document.getElementById('featureBars');
                if (!container || data.error) { if (container) container.innerHTML = '<div style="text-align:center;padding:0.75rem;color:var(--text-muted);font-size:0.8rem;">Feature data not available</div>'; return; }
                var features = data.features;
                var maxVal = Math.max.apply(null, features.map(function(f) { return Math.abs(f.value); }).concat([0.001]));
                var fhtml = '';
                features.forEach(function(f) {
                    var displayVal = f.value >= 1000 ? f.value.toExponential(1) : (Number.isInteger(f.value) ? f.value : f.value.toFixed(3));
                    fhtml += '<div class="feature-bar-row">' +
                        '<div class="feature-bar-label" title="' + f.raw_name + '">' + f.name + '</div>' +
                        '<div class="feature-bar-track"><div class="feature-bar-fill" style="width:0%"></div></div>' +
                        '<div class="feature-bar-val">' + displayVal + '</div></div>';
                });
                container.innerHTML = fhtml;
                setTimeout(function() {
                    container.querySelectorAll('.feature-bar-fill').forEach(function(bar, i) {
                        var pct = (Math.abs(features[i].value) / maxVal * 100).toFixed(1);
                        bar.style.width = pct + '%';
                    });
                }, 50);
            })
            .catch(function() {
                var container = document.getElementById('featureBars');
                if (container) container.innerHTML = '<div style="text-align:center;padding:0.75rem;color:var(--text-muted);font-size:0.8rem;">Feature data not available</div>';
            });

        overlay.classList.add('open');
        panel.classList.add('open');
        document.getElementById('detailClose').focus();
    }

    /* ================================================================
       AI Advisor Chat
       ================================================================ */
    let chatAddress = null;
    let chatHistory = [];

    function toggleChat(addr) {
        const section = document.getElementById('chatSection');
        if (!section) return;

        if (chatAddress !== addr) {
            chatAddress = addr;
            chatHistory = [];
            document.getElementById('chatMessages').innerHTML =
                '<div class="chat-bubble ai">Hello! I can help you understand why this address was flagged and what the fraud indicators mean. What would you like to know?</div>';
        }

        section.classList.toggle('open');
        if (section.classList.contains('open')) {
            const inp = document.getElementById('chatInput');
            if (inp) inp.focus();
        }
    }

    function escapeHtml(s) {
        return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    function formatResponse(text) {
        var s = escapeHtml(text);
        s = s.replace(RegExp('[*][*](.*?)[*][*]', 'g'), '<strong>$1</strong>');
        s = s.replace(RegExp('\\n', 'g'), '<br>');
        return s;
    }

    async function sendChat() {
        const input = document.getElementById('chatInput');
        const msg = input.value.trim();
        if (!msg || !chatAddress) return;
        input.value = '';

        const messages = document.getElementById('chatMessages');

        // User bubble
        messages.innerHTML += '<div class="chat-bubble user">' + escapeHtml(msg) + '</div>';
        messages.scrollTop = messages.scrollHeight;

        // Loading indicator
        const loadId = 'ld-' + Date.now();
        messages.innerHTML += '<div class="chat-bubble ai" id="' + loadId + '"><em>Thinking\u2026</em></div>';
        messages.scrollTop = messages.scrollHeight;

        try {
            const resp = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    address: chatAddress,
                    message: msg,
                    conversation_history: chatHistory
                })
            });
            const data = await resp.json();

            const ld = document.getElementById(loadId);
            if (ld) ld.remove();

            if (data.error) {
                messages.innerHTML += '<div class="chat-bubble ai" style="color:var(--danger);">Error: ' + escapeHtml(data.error) + '</div>';
            } else {
                chatHistory.push({role: "user", content: msg});
                chatHistory.push({role: "model", content: data.response});
                messages.innerHTML += '<div class="chat-bubble ai">' + formatResponse(data.response) + '</div>';
            }
        } catch (err) {
            const ld = document.getElementById(loadId);
            if (ld) ld.remove();
            messages.innerHTML += '<div class="chat-bubble ai" style="color:var(--danger);">Network error: ' + escapeHtml(err.message) + '</div>';
        }
        messages.scrollTop = messages.scrollHeight;
    }

    /* ════════════════════════════════════════════════════════════
       Mobile-only AI Advisor (floating FAB + bottom-sheet chat)
       Reuses chatAddress / chatHistory from the desktop chat above.
       ════════════════════════════════════════════════════════════ */
    function openMobileChat() {
        const panel = document.getElementById('mobileChatPanel');
        const messages = document.getElementById('mobileChatMessages');
        const ctx = document.getElementById('mcContext');
        const picker = document.getElementById('mcAddrPicker');
        const inputBar = document.getElementById('mcInputBar');
        if (!panel) return;

        if (chatAddress) {
            ctx.textContent = 'About: ' + chatAddress;
            picker.classList.remove('show');
            inputBar.style.display = '';
            if (!messages.innerHTML.trim()) {
                messages.innerHTML = '<div class="chat-bubble ai">Hello! I can help you understand why this address was flagged. What would you like to know?</div>';
            }
            setTimeout(function(){ document.getElementById('mobileChatInput')?.focus(); }, 50);
        } else {
            ctx.textContent = 'Pick an address to analyse';
            picker.classList.add('show');
            inputBar.style.display = 'none';
            messages.innerHTML = '';
            setTimeout(function(){ document.getElementById('mcAddrInput')?.focus(); }, 50);
        }
        panel.classList.add('open');
    }

    async function mcAnalyseAndChat() {
        const inp = document.getElementById('mcAddrInput');
        const btn = document.getElementById('mcAnalyseBtn');
        const messages = document.getElementById('mobileChatMessages');
        const picker = document.getElementById('mcAddrPicker');
        const inputBar = document.getElementById('mcInputBar');
        const ctx = document.getElementById('mcContext');

        let addr = (inp.value || '').trim();
        if (!/^0x[a-fA-F0-9]{40}$/.test(addr)) {
            messages.innerHTML = '<div class="chat-bubble ai" style="color:var(--danger);">That doesn\'t look like a valid Ethereum address. It should start with <strong>0x</strong> and be 42 characters long.</div>';
            return;
        }

        btn.disabled = true;
        btn.textContent = 'Analysing\u2026';
        messages.innerHTML = '<div class="chat-bubble ai"><em>Pulling transaction history and scoring this address\u2026 this can take 10-30 seconds.</em></div>';

        try {
            const resp = await fetch('/api/score/' + addr);
            const data = await resp.json();
            if (data.error) {
                messages.innerHTML = '<div class="chat-bubble ai" style="color:var(--danger);">Error: ' + escapeHtml(data.error) + '</div>';
                return;
            }

            // Switch into chat mode
            chatAddress = addr;
            chatHistory = [];
            picker.classList.remove('show');
            inputBar.style.display = '';
            ctx.textContent = 'About: ' + addr;

            const risk = data.risk_level || 'UNKNOWN';
            const prob = data.fraud_probability != null ? (data.fraud_probability * 100).toFixed(1) + '%' : '';
            messages.innerHTML =
                '<div class="chat-bubble ai">Done. This address scores <strong>' + prob + '</strong> &mdash; risk level <strong>' + escapeHtml(risk) + '</strong>. Ask me anything about why.</div>';
            setTimeout(function(){ document.getElementById('mobileChatInput')?.focus(); }, 50);
        } catch (err) {
            messages.innerHTML = '<div class="chat-bubble ai" style="color:var(--danger);">Network error: ' + escapeHtml(err.message) + '</div>';
        } finally {
            btn.disabled = false;
            btn.textContent = 'Analyse & chat';
        }
    }

    function closeMobileChat() {
        document.getElementById('mobileChatPanel')?.classList.remove('open');
    }

    async function sendMobileChat() {
        const input = document.getElementById('mobileChatInput');
        const messages = document.getElementById('mobileChatMessages');
        const msg = input.value.trim();
        if (!msg) return;
        if (!chatAddress) {
            messages.innerHTML += '<div class="chat-bubble ai" style="color:var(--danger);">Analyse an address first so I have something to discuss.</div>';
            return;
        }
        input.value = '';
        messages.innerHTML += '<div class="chat-bubble user">' + escapeHtml(msg) + '</div>';
        messages.scrollTop = messages.scrollHeight;

        const loadId = 'mld-' + Date.now();
        messages.innerHTML += '<div class="chat-bubble ai" id="' + loadId + '"><em>Thinking\u2026</em></div>';
        messages.scrollTop = messages.scrollHeight;

        try {
            const resp = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ address: chatAddress, message: msg, conversation_history: chatHistory })
            });
            const data = await resp.json();
            document.getElementById(loadId)?.remove();
            if (data.error) {
                messages.innerHTML += '<div class="chat-bubble ai" style="color:var(--danger);">Error: ' + escapeHtml(data.error) + '</div>';
            } else {
                messages.innerHTML += '<div class="chat-bubble ai">' + formatResponse(data.response) + '</div>';
                chatHistory.push({role: 'user', content: msg});
                chatHistory.push({role: 'model', content: data.response});
            }
        } catch (err) {
            document.getElementById(loadId)?.remove();
            messages.innerHTML += '<div class="chat-bubble ai" style="color:var(--danger);">Network error: ' + escapeHtml(err.message) + '</div>';
        }
        messages.scrollTop = messages.scrollHeight;
    }

    </script>

    <!-- ── Keyboard shortcuts help overlay ── -->
    <div class="shortcut-overlay" id="shortcutOverlay" onclick="if(event.target===this)this.classList.remove('open')">
        <div class="shortcut-modal">
            <h3>Keyboard Shortcuts</h3>
            <div class="shortcut-row"><span>Focus search</span><span class="shortcut-key">/</span></div>
            <div class="shortcut-row"><span>Next row</span><span class="shortcut-key">j</span></div>
            <div class="shortcut-row"><span>Previous row</span><span class="shortcut-key">k</span></div>
            <div class="shortcut-row"><span>Open detail</span><span class="shortcut-key">Enter</span></div>
            <div class="shortcut-row"><span>Copy address</span><span class="shortcut-key">c</span></div>
            <div class="shortcut-row"><span>Toggle dark mode</span><span class="shortcut-key">d</span></div>
            <div class="shortcut-row"><span>Close panel</span><span class="shortcut-key">Esc</span></div>
            <div class="shortcut-row"><span>Show shortcuts</span><span class="shortcut-key">?</span></div>
        </div>
    </div>
</body>
</html>
"""


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def home():
    """Render the web interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/score/<address>')
def score_address(address):
    """
    API endpoint to score an Ethereum address.
    
    Returns JSON with fraud probability and risk level.
    """
    try:
        result = detector.score_address(address)
        # Cache features for AI advisor, strip from response
        features = result.pop("features", None)
        if features:
            _cache_score(address, features, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "address": address}), 500


@app.route('/api/features/<address>')
def get_features(address):
    """Return cached feature values for a scored address."""
    address = address.strip().lower()
    cached = _get_cached(address)
    if not cached or 'features' not in cached:
        return jsonify({'error': 'Address not scored yet'}), 404

    features = cached['features']

    feature_labels = {
        'Avg min between sent tnx': 'Avg Time Between Sent Txns',
        'Avg min between received tnx': 'Avg Time Between Recv Txns',
        'Time Diff between first and last (Mins)': 'Account Age (Minutes)',
        'Sent tnx': 'Sent Transaction Count',
        'Received Tnx': 'Received Transaction Count',
        'Unique Received From Addresses': 'Unique Senders',
        'Unique Sent To Addresses': 'Unique Recipients',
        'avg val received': 'Avg ETH Received',
        'avg val sent': 'Avg ETH Sent',
        'total Ether sent': 'Total ETH Sent',
        'total ether received': 'Total ETH Received',
        'total ether balance': 'ETH Balance',
        ' ERC20 total Ether received': 'ERC20 ETH Received',
        ' ERC20 total ether sent': 'ERC20 ETH Sent',
        ' ERC20 uniq sent addr': 'ERC20 Unique Recipients',
        ' ERC20 uniq rec addr': 'ERC20 Unique Senders',
        'max val sent to contract': 'Max Sent to Contract',
        'total transactions (including tnx to create contract': 'Total Transactions',
        ' ERC20 total Ether sent contract': 'ERC20 Contract Transfers',
        'Number of Created Contracts': 'Contracts Created',
        'min value received': 'Min ETH Received',
        'max value received ': 'Max ETH Received',
        'min val sent': 'Min ETH Sent',
        'max val sent': 'Max ETH Sent',
        'min value sent to contract': 'Min Sent to Contract',
        ' Total ERC20 tnxs': 'Total ERC20 Transactions',
        ' ERC20 avg time between sent tnx': 'ERC20 Avg Time Sent',
        ' ERC20 avg time between rec tnx': 'ERC20 Avg Time Received',
        ' ERC20 avg time between rec 2 tnx': 'ERC20 Avg Time Rec2',
        ' ERC20 avg time between contract tnx': 'ERC20 Avg Contract Interval',
        ' ERC20 min val rec': 'ERC20 Min Received',
        ' ERC20 max val rec': 'ERC20 Max Received',
        ' ERC20 avg val rec': 'ERC20 Avg Received',
        ' ERC20 min val sent': 'ERC20 Min Sent',
        ' ERC20 max val sent': 'ERC20 Max Sent',
        ' ERC20 avg val sent': 'ERC20 Avg Sent',
        ' ERC20 uniq sent token name': 'ERC20 Unique Tokens Sent',
        ' ERC20 uniq rec token name': 'ERC20 Unique Tokens Received',
        'avg_val_sent_contract': 'Avg Sent to Contract',
        'log_ether_sent': 'Log ETH Sent',
        'decile': 'Account Age Decile',
    }

    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    result = [{'name': feature_labels.get(n, n), 'raw_name': n, 'value': v} for n, v in sorted_features]
    return jsonify({'features': result, 'address': address})


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": detector.models_loaded,
        "stream_active": stream_monitor is not None and stream_monitor._running,
    })


# ============================================================================
# Stream endpoints
# ============================================================================

@app.route('/api/stream')
def stream_sse():
    """SSE endpoint. Pushes every scored address to the browser as it lands."""
    from queue import Queue
    q = Queue(maxsize=200)
    stream_clients.append(q)

    def generate():
        try:
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg  # already formatted as "event: ...\ndata: ...\n\n"
                except Exception:
                    yield ": keepalive\n\n"
        finally:
            if q in stream_clients:
                stream_clients.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route('/api/stream/stats')
def stream_stats():
    """Return current stream monitor stats."""
    if stream_monitor is None:
        return jsonify({"error": "Stream monitor not running"}), 503
    return jsonify(stream_monitor.stats)


@app.route('/api/stream/alerts')
def stream_alerts():
    """Return recent high-risk alerts."""
    if stream_monitor is None:
        return jsonify({"error": "Stream monitor not running"}), 503
    return jsonify(stream_monitor.recent_alerts)


# ============================================================================
# AI Advisor Chat
# ============================================================================

def _build_system_prompt(address, features, score_result):
    """Build a detailed system prompt for the Gemini AI advisor."""
    prob = score_result.get("fraud_probability") or 0
    risk = score_result.get("risk_level") or "UNKNOWN"
    ms = score_result.get("model_scores") or {}
    stats = score_result.get("address_stats") or {}
    method = score_result.get("ensemble_method") or "weighted_average"
    active_mode = score_result.get("scoring_mode") or "fixed"
    ada_weights = score_result.get("adaptive_weights")

    feature_text = "\n".join(f"  - {name}: {value}" for name, value in features.items())

    # Build dynamic architecture description
    if method == "adaptive_weighted_3model":
        w_desc = ""
        if ada_weights:
            w_desc = (f"\nPer-address weights for THIS address: w_XGB={ada_weights['w_xgb']:.3f}, "
                      f"w_RF={ada_weights['w_rf']:.3f}, w_IF={ada_weights['w_if']:.3f}")
        arch_desc = f"""Ensemble method: Adaptive Weighted Ensemble (Tier 2A) — fuzzy-logic-inspired per-address weighting.
Active mode: ADAPTIVE{w_desc}

HOW ADAPTIVE WEIGHTING WORKS:
Instead of a fixed 5% weight for the Isolation Forest on every address, the system computes a per-address IF weight (between 2% and 30%) using smooth sigmoid membership functions inspired by fuzzy logic (but implemented without a fuzzy inference library).
Two antecedents drive the weight adjustment:
1. IF anomaly strength: How strongly the IF flags this address as anomalous (sigmoid centred at 0.6).
2. Supervised consensus: The weighted average of XGB and RF scores (0.190*XGB + 0.810*RF). When this is low (<0.15), supervised models are uncertain and IF evidence becomes more valuable.

Four rules fire with smooth activation (no hard thresholds):
- R1 AMPLIFICATION: IF high + supervised low → boost IF weight up to +25pp (catches exploits missed by supervised models).
- R2 DAMPENING: IF high + supervised NOT low → reduce IF weight by up to 3pp (protects legitimate high-volume contracts like Uniswap V2 Router where supervised models correctly assign moderate scores).
- R3 TIEBREAKER: Supervised models disagree (|XGB−RF| > 0.2) + IF high → small +5pp boost to let IF break the tie.
- R4 IMPLICIT: When IF anomaly is low, no rules activate strongly and the weight stays near the 5% base.

After computing the raw IF weight, XGB and RF weights are redistributed proportionally from the remaining (1 − w_IF), preserving their 19:81 ratio from Perrone & Cooper (1993).

INTERPRETING THE PER-ADDRESS WEIGHTS:
- w_IF near 0.05 (base): IF anomaly is low or supervised models are confident — standard behaviour, same as fixed mode.
- w_IF >> 0.05 (e.g. 0.20–0.30): IF detected a strong anomaly AND supervised models gave low scores — the system is amplifying the unsupervised signal. This typically happens for exploit addresses where XGB≈0 and RF is low but IF flags unusual transaction structure.
- w_IF << 0.05 (e.g. 0.02–0.04): IF anomaly is high but supervised models are moderately confident — dampening protects against false positives on legitimate contracts.

EVALUATION RESULTS (ADAPTIVE vs FIXED):
- Holdout set (1,477 addresses): AUC-ROC 0.9968 (adaptive) vs 0.9983 (fixed) — minimal difference, both excellent.
- Known fraud recall (Test 3, 12 real exploit addresses): 91.7% adaptive vs 16.7% fixed — massive improvement. Adaptive detects 11/12 known exploits including Ronin Bridge, Euler Finance, Wintermute, Cream Finance. The one miss is due to missing Etherscan data.
- Known legitimate false positives (Test 4, 20 entities): 85% FP rate adaptive vs 5% fixed — significant trade-off. High-volume legitimate addresses (Binance Hot Wallet, Coinbase) trigger the amplification rule because their supervised scores are also very low (XGB≈0, RF<0.1), making them indistinguishable from exploits to the rule engine.
- Uniswap V2 Router (Test 6): Protected by dampening — score only +0.016 above fixed (0.317 vs 0.301) because its supervised consensus of 0.264 triggers R2.
- Gnosis Safe (Test 6): Unaffected — IF score of 0.133 is too low to trigger any rules, score identical to fixed.

This is an inherent limitation: addresses with both low supervised scores AND high IF anomaly (common for both real exploits and legitimate high-volume contracts) cannot be distinguished by the rule engine alone. The adaptive mode is best understood as an experimental enhancement that trades FP precision for exploit recall.

The base fixed-weight formula (for comparison) is: 0.95 * (0.190*XGB + 0.810*RF) + 0.05 * IF"""
    elif method == "weighted_average_3model":
        arch_desc = """Ensemble method: 3-Model Weighted Ensemble (Tier 2) combining supervised and unsupervised approaches.
Active mode: FIXED (constant weights per address).
Formula: final = 0.95 * (0.190*XGB + 0.810*RF) + 0.05 * IF
The two supervised models (XGBoost and Random Forest) carry 95% of the weight, with inverse-error weighting (Perrone & Cooper, 1993) based on validation AUC-ROC.
The Isolation Forest (unsupervised anomaly detector, 5% weight) detects structural anomalies in transaction patterns without using fraud labels.
The IF weight of 0.05 was empirically validated on the holdout set to maximize F1-score.
The F1-optimal threshold is 0.27, achieving F1=0.9803, precision=0.9900, recall=0.9707 on the 1,477-address holdout set (AUC-ROC=0.9983, AUC-PR=0.9977).

ADAPTIVE MODE (available via toggle): The system also supports an adaptive weighting mode (Tier 2A) that adjusts the IF weight per address from 2% to 30% using fuzzy-logic-inspired sigmoid rules. It dramatically improves recall on known exploit addresses (16.7% → 91.7%) but at the cost of higher false positives on legitimate entities (5% → 85%). The user can switch between modes using the toggle in the header or ask about the differences. This address was scored under FIXED mode.

STACKER EXPERIMENT (Tier 1 — disabled): A LightGBM meta-learner was trained as a stacked generalisation layer (Wolpert, 1992) on the calibrated outputs of all three base models. While it improved holdout metrics (AUC-ROC 0.9992, F1 0.9877 at threshold 0.58), it was disabled in production after evaluation revealed critical overfitting on live data: 0% recall on 12 known fraud addresses, and extreme false positives on edge cases (Gnosis Safe Multi-Sig scored 0.97 CRITICAL vs 0.27 MINIMAL under weighted average). The stacker is discussed as a failed experiment in the dissertation (Section 4.7)."""
    elif method == "weighted_average_2model":
        arch_desc = """Ensemble method: 2-Model Weighted Average (Perrone & Cooper, 1993).
The final probability is a linear combination of the two supervised models, with weights derived analytically from their validation error:
1. Random Forest: 81.0% weight (0.9981 AUC-ROC on validation set).
2. XGBoost (calibrated): 19.0% weight (0.9919 AUC-ROC, calibrated via isotonic regression).
The Isolation Forest was unavailable for this scoring, so only supervised models were used."""
    else:  # xgboost_only
        arch_desc = """Current ensemble method: Single Model (XGBoost Only).
The system fell back to this because other models (Random Forest, Isolation Forest) were unavailable.
The final probability is derived solely from the XGBoost model, calibrated via isotonic regression."""

    return f"""You are a fraud analysis advisor for an Ethereum blockchain fraud detection system built as a B.Eng. thesis project at the University of Malta.

STRICT RULES:
- You may ONLY discuss topics related to Ethereum fraud detection, this specific address's analysis, blockchain security, and the detection methodology used.
- If the user asks about anything unrelated to fraud detection or this address, politely decline and redirect them to ask about the fraud analysis.
- Never fabricate data. Only reference the feature values and scores provided below.
- Do not reference external data. All your analysis must be based solely on the data provided in this prompt.
- Keep responses SHORT — 2-3 sentences max. Be direct and specific. No filler or preamble.
- Put a blank line between each sentence (use two newlines). This is mandatory for readability.

DETECTION SYSTEM ARCHITECTURE:
{arch_desc}

RISK THRESHOLDS:
- CRITICAL: >= 90% fraud probability
- HIGH: >= 70%
- MEDIUM: >= 50%
- LOW: >= 30%
- MINIMAL: < 30%

CURRENT ADDRESS ANALYSIS:
Address: {address}
Final Ensemble Fraud Probability: {prob:.4f} ({prob*100:.1f}%)
Risk Level: {risk}
Ensemble Method Used: {method.replace('_', ' ').title()}

Individual Model Scores:
- XGBoost (calibrated, supervised): {ms.get('xgboost', 'N/A')}
- Random Forest (calibrated, supervised): {ms.get('random_forest', 'N/A')}
- Isolation Forest (unsupervised anomaly score): {ms.get('isolation_forest', 'N/A')}

Address Statistics:
- Total transactions: {stats.get('total_transactions', 'N/A')}
- ERC-20 transfers: {stats.get('total_erc20_transfers', 'N/A')}
- ETH balance: {stats.get('balance_eth', 'N/A')} ETH
- Total ETH sent: {stats.get('total_ether_sent', 'N/A')} ETH
- Total ETH received: {stats.get('total_ether_received', 'N/A')} ETH

FULL 48-FEATURE VECTOR (these are the features computed from on-chain data and fed to the models):
{feature_text}

KEY FEATURE EXPLANATIONS:
- "Avg min between sent/received tnx": Average time between transactions. Very low values can indicate automated/bot activity.
- "Unique Received From / Sent To Addresses": Low unique counterparties with high volume can suggest wash trading or mixer usage.
- "Number of Created Contracts": Fraudulent addresses sometimes deploy multiple contracts.
- "total Ether sent/received": Large imbalances can indicate fund draining or accumulation schemes.
- "Total ERC20 tnxs": High ERC-20 activity with low normal transactions can indicate token-related scams.
- "log_ether_sent": Log-transformed Ether sent — helps normalize extreme values.
- "decile": Address lifespan decile (0=oldest, 9=newest). New addresses are statistically riskier.
- Velocity features (tx_count_last_*m, ether_sent_last_*m): Activity in recent time windows. Spikes indicate sudden activity bursts.
- Ratio features (ratio_vs_avg_*m): Current activity compared to historical averages. High ratios indicate abnormal spikes.

TIERED ENSEMBLE ARCHITECTURE (for context if the user asks about the system):
- Tier 1: Stacked Generalisation (LightGBM meta-learner) — DISABLED due to overfitting on live data.
- Tier 2: Fixed 3-Model Weighted Average — production default. Constant weights: 0.190*XGB + 0.810*RF (95%), 0.05*IF (5%).
- Tier 2A: Adaptive 3-Model Weighted Ensemble — experimental. Per-address IF weight from 2% to 30% via fuzzy-logic-inspired rules.
- Tier 3: 2-Model Weighted Average (XGB + RF only) — fallback when IF is unavailable.
- Tier 4: XGBoost Only — last-resort fallback.
The system automatically degrades through tiers based on which models successfully produce scores.

UI FEATURES THE USER MAY ASK ABOUT:
- Mode Toggle: The button in the header switches between Fixed and Adaptive modes at runtime. It affects all subsequent scoring requests without a server restart.
- Compare Modes: In the detail panel for any scored address, a "Compare Modes" button scores the same address under both Fixed and Adaptive modes side-by-side using cached base-model scores (no extra API calls). This lets users see exactly how the adaptive weights change the final score.
- Per-address weights: When in adaptive mode, the weights (w_XGB, w_RF, w_IF) are displayed beneath the score. These are unique to each address.

IMPORTANT CONTEXT — ISOLATION FOREST BEHAVIOUR:
The Isolation Forest has an AUC-ROC of only 0.3517 on the holdout set, meaning it is inversely correlated with fraud: fraudulent addresses tend to have HIGH IF scores (close to 1.0) while legitimate addresses have MODERATE IF scores. This is counterintuitive but expected — the IF detects structural anomalies in transaction patterns, and fraud addresses often look "normal" to an anomaly detector (e.g., simple fund drains), while complex legitimate DeFi contracts look "anomalous." The adaptive weighting rules are designed around this inverse correlation.

When answering questions:
1. Be brief — answer in 2-3 sentences. No bullet lists unless asked.
2. Reference specific feature values and model scores to justify your answer.
3. Use plain language but be precise about the numbers.
4. If models disagree, explain briefly what each one is detecting.
5. If the system is using a fallback method (Weighted Average or XGBoost Only), briefly explain why if the user asks about the methodology.
6. If the user asks about adaptive vs fixed mode, explain the trade-off honestly: adaptive dramatically improves exploit detection but at the cost of more false positives on legitimate addresses.
7. If the user asks why the IF weight is high or low for their address, explain it in terms of the two antecedents (IF anomaly strength and supervised consensus).
8. If the user asks about the per-address weights displayed, explain what each weight means and why they differ from the fixed baseline."""


@app.route('/api/chat', methods=['POST'])
def chat():
    """AI Fraud Advisor chat endpoint using Google Gemini."""
    data = request.get_json()
    address = (data.get("address") or "").strip().lower()
    message = (data.get("message") or "").strip()
    history = data.get("conversation_history", [])

    if not address or not message:
        return jsonify({"error": "Address and message are required."}), 400

    try:
        # Get cached features + score, or re-fetch
        cached = _get_cached(address)
        if cached:
            features = cached["features"]
            score_result = cached["result"]
        else:
            result = detector.score_address(address)
            features = result.pop("features", {})
            score_result = result
            if features:
                _cache_score(address, features, score_result)

        system_prompt = _build_system_prompt(address, features, score_result)

        model = genai.GenerativeModel(
            model_name="gemini-3.1-flash-lite",
            system_instruction=system_prompt,
        )

        # Convert history to Gemini format
        gemini_history = []
        for msg in history:
            gemini_history.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [msg["content"]],
            })

        chat_session = model.start_chat(history=gemini_history)

        response = chat_session.send_message(message)
        return jsonify({"response": response.text})
    except Exception as e:
        print(f"AI Advisor Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"AI advisor error: {str(e)}"}), 500


# ============================================================================
# Scoring Mode API
# ============================================================================

@app.route('/api/scoring-mode', methods=['GET'])
def get_scoring_mode():
    """Return the current scoring mode."""
    return jsonify({"mode": scoring_config.mode})


@app.route('/api/scoring-mode', methods=['POST'])
def set_scoring_mode():
    """Switch the active scoring mode at runtime (no restart required)."""
    data = request.get_json()
    new_mode = (data or {}).get("mode", "").strip().lower()
    try:
        scoring_config.mode = new_mode
        return jsonify({"mode": scoring_config.mode})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/compare/<address>')
def compare_modes(address):
    """Score an address under both fixed and adaptive modes side-by-side."""
    try:
        result = detector.score_address_compare(address)
        # Strip features from both payloads, cache from fixed result
        for key in ("fixed", "adaptive"):
            features = result[key].pop("features", None)
            if features and key == "fixed":
                _cache_score(address, features, result[key])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "address": address}), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Fraud Detection Web Server")
    parser.add_argument('--port', type=int, default=5001, help='Port to run on')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Starting Fraud Detection Web Server")
    print(f"{'='*60}")
    print(f"URL:  http://{args.host}:{args.port}/")
    print(f"API:  http://{args.host}:{args.port}/api/score/<address>")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
