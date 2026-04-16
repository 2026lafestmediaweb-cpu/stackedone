import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
import google.generativeai as genai
import requests

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = (BASE_DIR.parent / "externship").resolve()

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
conversation_history = []


def get_gemini_api_key() -> str:
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
        or ""
    ).strip()


def get_model_candidates() -> list[str]:
    preferred = os.getenv("GEMINI_MODEL", "").strip()
    fallbacks = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash-latest",
    ]
    if preferred:
        return [preferred, *[m for m in fallbacks if m != preferred]]
    return fallbacks


def get_supabase_config() -> dict:
    return {
        "url": os.getenv("SUPABASE_URL", "").strip(),
        "key": os.getenv("API_KEY", "").strip(),
        "table": os.getenv("SUPABASE_CHAT_TABLE", "chat_messages").strip(),
    }


def save_message_to_db(role: str, text: str) -> None:
    config = get_supabase_config()
    if not config["url"] or not config["key"]:
        return

    endpoint = f"{config['url'].rstrip('/')}/rest/v1/{config['table']}"
    headers = {
        "apikey": config["key"],
        "Authorization": f"Bearer {config['key']}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    payload = {"role": role, "text": text}
    response = requests.post(endpoint, headers=headers, json=payload, timeout=20)
    response.raise_for_status()


def fetch_messages_from_db(limit: int) -> list[dict]:
    config = get_supabase_config()
    if not config["url"] or not config["key"]:
        return []

    endpoint = f"{config['url'].rstrip('/')}/rest/v1/{config['table']}"
    headers = {
        "apikey": config["key"],
        "Authorization": f"Bearer {config['key']}",
    }
    params = {
        "select": "id,role,text,created_at",
        "order": "id.asc",
        "limit": max(1, min(limit, 500)),
    }
    response = requests.get(endpoint, headers=headers, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    user_text = data.get("text")

    if not isinstance(user_text, str) or not user_text.strip():
        return jsonify({"error": "Please provide non-empty 'text' in JSON body."}), 400

    api_key = get_gemini_api_key()
    if not api_key:
        return jsonify({"error": "Gemini API key not found in backend .env file."}), 500

    user_text = user_text.strip()
    conversation_history.append({"role": "user", "text": user_text})
    try:
        save_message_to_db("user", user_text)
    except Exception:
        # Keep chat responsive even if DB write fails.
        pass

    try:
        genai.configure(api_key=api_key)
        ai_text = ""
        last_error = None

        for model_name in get_model_candidates():
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(user_text)
                ai_text = (response.text or "").strip() or "I could not generate a response."
                break
            except Exception as model_error:
                last_error = model_error

        if not ai_text:
            raise RuntimeError(
                f"None of the configured Gemini models worked. Last error: {last_error}"
            )
    except Exception as error:
        return jsonify({"error": f"Gemini request failed: {error}"}), 500

    conversation_history.append({"role": "ai", "text": ai_text})
    try:
        save_message_to_db("ai", ai_text)
    except Exception:
        # Keep chat responsive even if DB write fails.
        pass

    return jsonify({"reply": ai_text, "conversation": conversation_history}), 200


@app.route("/chat/messages", methods=["GET"])
def get_chat_messages():
    limit_raw = request.args.get("limit", "100")
    try:
        limit = int(limit_raw)
    except ValueError:
        return jsonify({"error": "Query param 'limit' must be an integer."}), 400

    config = get_supabase_config()
    if not config["url"] or not config["key"]:
        return jsonify(
            {
                "error": (
                    "Database is not configured. Set SUPABASE_URL and API_KEY in .env, "
                    "then create a table (default: chat_messages)."
                )
            }
        ), 500

    try:
        messages = fetch_messages_from_db(limit)
    except Exception as error:
        return jsonify({"error": f"Failed to fetch messages from database: {error}"}), 500

    return jsonify({"messages": messages}), 200


if __name__ == "__main__":
    app.run(debug=True)
