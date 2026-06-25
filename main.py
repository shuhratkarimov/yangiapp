from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import os
import shutil
import uuid
import re
import time
import random
from collections import Counter

import fitz  # PyMuPDF
import genanki
import json
import requests

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Anki-Tan API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "images"), exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ╔═══════════════════════════════════════════════════════════════╗
# ║  API KALITNI SHU YERGA QOYING                                ║
# ║  https://aistudio.google.com dan oling                       ║
# ╚═══════════════════════════════════════════════════════════════╝
API_KEY = os.environ.get("GEMINI_API_KEY", "")

MODEL_NAME = "gemini-2.0-flash"

LANG_NAMES = {
    "en": "English", "ru": "Russian", "uz": "Uzbek",
    "de": "German", "fr": "French", "es": "Spanish",
    "ar": "Arabic", "tr": "Turkish", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "it": "Italian",
    "pt": "Portuguese", "hi": "Hindi", "fa": "Persian",
}

# ── Anki karta shablon (Colab dan aynan ko'chirilgan) ──

CARD_CSS = (
    ".card{font-family:'Segoe UI',Arial,sans-serif;font-size:18px;"
    "text-align:center;color:#2c3e50;background:#fafafa;padding:20px}"
    ".term{font-size:26px;font-weight:bold;color:#1a5276;margin-bottom:10px}"
    ".badge{display:inline-block;padding:2px 10px;border-radius:10px;"
    "font-size:11px;font-weight:600;color:#fff;margin-bottom:12px}"
    ".sci{background:#c0392b}.b1{background:#27ae60}.col{background:#2980b9}"
    ".ex{font-style:italic;color:#7f8c8d;font-size:14px;margin-top:8px}"
    ".def{font-size:16px;text-align:left;margin-top:8px;line-height:1.5}"
    ".tr{font-size:22px;font-weight:bold;color:#16a085;margin:12px 0}"
    ".pg{font-size:10px;color:#bbb;margin-top:8px}hr{border:1px solid #eee}"
)

CARD_FRONT = (
    '<div class="term">{{Term}}</div>'
    '<span class="badge {{BadgeCSS}}">{{Category}}</span>'
    '<div class="ex">{{Example}}</div>'
)

CARD_BACK = (
    '{{FrontSide}}<hr>'
    '<div class="tr">{{Translation}}</div>'
    '<div class="def">{{Definition}}</div>'
    '<div class="pg">{{Page}}</div>'
)

CATEGORY_MAP = {
    "scientific_term": ("sci", "Scientific Term"),
    "b1_plus_word": ("b1", "B1+ Vocabulary"),
    "collocation": ("col", "Collocation"),
}

MAX_CHUNK = 18000


# ── Colab dan olingan funksiyalar ──

def fix_truncated_json(raw):
    raw = raw.strip()
    if raw.endswith("]"):
        return raw
    last_brace = raw.rfind("}")
    if last_brace < 0:
        return None
    return raw[:last_brace + 1].rstrip().rstrip(",") + "\n]"


def call_gemini_for_terms(text_chunk, src, tgt, mono=False):
    if mono:
        trans_rule = "Set translation to empty string. Write definition in " + src + "."
    else:
        trans_rule = "Write translation in " + tgt + ". Write definition in " + src + "."

    prompt = (
        "You are a vocabulary extraction expert.\n"
        "Extract important vocabulary from the text below in 3 categories:\n"
        "1) scientific_term - technical domain terms\n"
        "2) b1_plus_word - general academic words B1+\n"
        "3) collocation - important 2-3 word phrases\n\n"
        "RULES:\n"
        "- 40-80 unique terms. No duplicates.\n"
        "- " + trans_rule + "\n"
        "- Short example sentence for each term.\n"
        "- Record page number from [PAGE X] markers.\n"
        "- Output ONLY a JSON array. No markdown. No fences. No explanation.\n\n"
        "Format per item:\n"
        '{"term":"...","category":"...","definition":"...","translation":"...","example":"...","page":0}\n\n'
        "TEXT:\n" + text_chunk
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {
            "parts": [{"text": "You output only valid JSON arrays. Never add markdown or explanations."}]
        },
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192,
        },
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()

            if "candidates" not in data or not data["candidates"]:
                logger.warning(f"No candidates (urinish {attempt+1})")
                time.sleep(5)
                continue

            raw = data["candidates"][0]["content"]["parts"][0]["text"]
            if not raw or not raw.strip():
                logger.warning(f"Bo'sh javob (urinish {attempt+1})")
                time.sleep(5)
                continue

            raw = raw.strip()
            logger.info(f"{len(raw)} belgi javob olindi")

            if raw.startswith("﻿"):
                raw = raw[1:]
            raw = re.sub(r"^```(?:json)?[\s\n]*", "", raw)
            raw = re.sub(r"[\s\n]*```$", "", raw)
            raw = raw.strip()

            fb = raw.find("[")
            if fb < 0:
                logger.warning(f"[ topilmadi (urinish {attempt+1})")
                time.sleep(5)
                continue
            raw = raw[fb:]

            lb = raw.rfind("]")
            if lb > 0:
                raw = raw[:lb+1]
            else:
                logger.warning("] topilmadi — kesilgan javob tuzatilmoqda...")
                fixed = fix_truncated_json(raw)
                if fixed:
                    raw = fixed
                    logger.info(f"Tuzatildi ({len(raw)} belgi)")
                else:
                    logger.warning(f"Tuzatib bo'lmadi (urinish {attempt+1})")
                    time.sleep(5)
                    continue

            terms = json.loads(raw)

            if isinstance(terms, list) and len(terms) > 0:
                logger.info(f"{len(terms)} ta termin topildi!")
                return terms
            else:
                logger.warning(f"Bo'sh natija (urinish {attempt+1})")
                time.sleep(5)

        except json.JSONDecodeError as je:
            logger.warning(f"JSON xato: {je} (urinish {attempt+1})")
            time.sleep(5)
        except requests.exceptions.HTTPError as e:
            logger.error(f"API HTTP xato: {e}")
            raise
        except Exception as ex:
            msg = str(ex)
            logger.warning(f"{type(ex).__name__}: {msg[:120]} (urinish {attempt+1})")
            time.sleep(5)

    return []


def extract_text_from_pdf(file_path, start, end):
    doc = fitz.open(file_path)
    total_pages = len(doc)
    s = max(0, start - 1)
    e = min(end, total_pages)

    page_texts = []
    for i in range(s, e):
        txt = doc[i].get_text("text").strip()
        if txt:
            page_texts.append(f"[PAGE {i + 1}]\n{txt}")
    doc.close()
    return page_texts


def chunk_text(page_texts):
    if not page_texts:
        return []
    full = "\n\n".join(page_texts)
    if len(full) <= MAX_CHUNK:
        return [full]

    chunks = []
    cur = ""
    for pt in page_texts:
        if len(cur) + len(pt) + 2 > MAX_CHUNK:
            if cur:
                chunks.append(cur)
            cur = pt
        else:
            cur = (cur + "\n\n" + pt) if cur else pt
    if cur:
        chunks.append(cur)
    return chunks


def generate_output_files(terms, session_id, deck_name, pdf_name, start_page, end_page, src, tgt):
    base_name = f"ankitan_{session_id}"

    mid = random.randrange(1 << 30, 1 << 31)
    did = random.randrange(1 << 30, 1 << 31)

    anki_model = genanki.Model(
        mid, "PDF_Vocab_v5",
        fields=[
            {"name": "Term"}, {"name": "BadgeCSS"}, {"name": "Category"},
            {"name": "Definition"}, {"name": "Translation"},
            {"name": "Example"}, {"name": "Page"},
        ],
        templates=[{"name": "Card 1", "qfmt": CARD_FRONT, "afmt": CARD_BACK}],
        css=CARD_CSS,
    )

    deck = genanki.Deck(did, deck_name)
    card_count = 0

    for t in terms:
        w = str(t.get("term", "")).strip()
        if not w:
            continue
        cat = t.get("category", "b1_plus_word")
        bc, cl = CATEGORY_MAP.get(cat, ("b1", str(cat)))
        pg = t.get("page")

        note = genanki.Note(
            model=anki_model,
            fields=[
                w, bc, cl,
                str(t.get("definition", "")),
                str(t.get("translation", "") or ""),
                str(t.get("example", "")),
                f"p.{pg}" if pg else "",
            ],
            tags=[cat],
        )
        deck.add_note(note)
        card_count += 1

    apkg_path = os.path.join(OUTPUT_DIR, f"{base_name}.apkg")
    genanki.Package(deck).write_to_file(apkg_path)

    # .txt (Colab formatida)
    txt_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for t in terms:
            w = str(t.get("term", "")).strip()
            if not w:
                continue
            d = str(t.get("definition", "")).replace("\t", " ").replace("\n", " ")
            tr = str(t.get("translation", "") or "").replace("\t", " ").replace("\n", " ")
            ex = str(t.get("example", "")).replace("\t", " ").replace("\n", " ")
            bk = f"<b>{tr}</b><br>{d}"
            if ex:
                bk += f"<br><br><i>{ex}</i>"
            f.write(f"{w}\t{bk}\n")

    # .json (Colab formatida)
    json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
    cats = Counter(t.get("category", "?") for t in terms)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "source": pdf_name,
                "pages": f"{start_page}-{end_page}",
                "source_lang": src,
                "target_lang": tgt,
                "model": MODEL_NAME,
                "total": len(terms),
            },
            "terms": terms,
        }, f, ensure_ascii=False, indent=2)

    return {
        "card_count": card_count,
        "categories": dict(cats),
        "files": {
            "apkg": f"/download/{base_name}.apkg",
            "txt": f"/download/{base_name}.txt",
            "json": f"/download/{base_name}.json",
        },
    }


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/generate")
async def handle_upload(
    file: UploadFile = File(...),
    from_page: int = Form(...),
    to_page: int = Form(...),
    source_lang: str = Form(...),
    target_lang: str = Form(...),
):
    session_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        page_texts = extract_text_from_pdf(file_path, from_page, to_page)
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return JSONResponse({"error": f"PDF xatolik: {str(e)}"}, status_code=400)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    if not page_texts:
        return JSONResponse({"error": "PDF dan matn topilmadi"}, status_code=400)

    if not API_KEY:
        return JSONResponse(
            {"error": "Server API kaliti sozlanmagan. Render'da GEMINI_API_KEY env o'zgaruvchisini qo'shing."},
            status_code=500,
        )

    src = LANG_NAMES.get(source_lang, source_lang)
    tgt = LANG_NAMES.get(target_lang, target_lang)

    chunks = chunk_text(page_texts)
    logger.info(f"{len(chunks)} bo'lak, {sum(len(c) for c in chunks):,} belgi")

    all_terms = []
    seen = set()

    try:
        for ci, chunk in enumerate(chunks):
            logger.info(f"Bo'lak {ci + 1}/{len(chunks)} ({len(chunk):,} belgi)")
            terms = call_gemini_for_terms(chunk, src, tgt)
            for t in terms:
                key = t.get("term", "").strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    all_terms.append(t)
            if ci < len(chunks) - 1:
                time.sleep(3)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in (400, 403):
            return JSONResponse({"error": "API kalit noto'g'ri yoki yaroqsiz"}, status_code=400)
        return JSONResponse({"error": f"Gemini API xatolik: {str(e)}"}, status_code=500)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return JSONResponse({"error": f"Flashcard yaratishda xatolik: {str(e)}"}, status_code=500)

    if not all_terms:
        return JSONResponse({"error": "Terminlar topilmadi. Boshqa sahifalarni sinab ko'ring."}, status_code=400)

    deck_name = file.filename.replace(".pdf", "").replace(".PDF", "") + "_Vocab"
    result = generate_output_files(
        all_terms, session_id, deck_name,
        file.filename, from_page, to_page, src, tgt,
    )

    cats = Counter(t.get("category", "?") for t in all_terms)
    logger.info(f"JAMI: {len(all_terms)} ta termin | {dict(cats)}")

    preview = []
    for t in all_terms[:8]:
        cat = t.get("category", "b1_plus_word")
        badge = CATEGORY_MAP.get(cat, ("b1", cat))[0]
        preview.append({
            "term": t.get("term", ""),
            "translation": t.get("translation", ""),
            "category": cat,
            "badge": badge,
            "definition": t.get("definition", ""),
        })

    return JSONResponse({
        "success": True,
        "card_count": result["card_count"],
        "categories": result["categories"],
        "files": result["files"],
        "preview": preview,
    })


@app.get("/download/{filename}")
async def download_file(filename: str):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, safe_name)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=safe_name)
    return JSONResponse({"error": "Fayl topilmadi"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
