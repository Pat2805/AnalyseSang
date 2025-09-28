from __future__ import annotations
import os, json, csv, hashlib, time, textwrap
from pathlib import Path
from typing import Dict, List, Iterable
import pdfplumber
import yaml

from agent1_extract.models import BilanResult
from agent1_extract.normalize import normalize_and_dedup
import traceback

# ---------------------- DEBUG/CONFIG ---------------------------------
DEBUG = True                      # passe à False quand tout roule
MAX_BLOCK_CHARS = 6000            # évite d'envoyer des pavés au LLM
LOG_DIR = Path("out/_debug")      # dépôt central des logs
LOG_DIR.mkdir(parents=True, exist_ok=True)

PDF_DIR   = Path("data/pdf")
OUT_DIR   = Path("out")
CACHE_DIR = OUT_DIR / ".cache"
INDEX_F   = OUT_DIR / "ingest_index.json"
FACTS_JL  = OUT_DIR / "facts.jsonl"
FACTS_CSV = OUT_DIR / "facts.csv"

PROMPTS_YAML = Path("agent1_extract/prompts.yaml")
UNITS_CFG    = Path("agent1_extract/units_config.json")
SETTINGS_JSON= Path("agent1_extract/settings.json")

# ---------------------- LOG HELPERS ----------------------------------
def log(msg: str):
    print(msg)
    with (LOG_DIR / "run.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")

def dump(name: str, content: str | dict | list):
    p = LOG_DIR / name
    if isinstance(content, (dict, list)):
        p.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        p.write_text(str(content), encoding="utf-8")

# ---------------------- IO HELPERS -----------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def file_sig(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"

def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def append_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))

# ---------------------- SETTINGS & PROMPTS ---------------------------
def load_prompts_and_settings():
    if not PROMPTS_YAML.exists(): raise FileNotFoundError(f"Manque: {PROMPTS_YAML}")
    if not UNITS_CFG.exists():    raise FileNotFoundError(f"Manque: {UNITS_CFG}")
    if not SETTINGS_JSON.exists():raise FileNotFoundError(f"Manque: {SETTINGS_JSON}")

    prompts  = load_yaml(PROMPTS_YAML)
    units    = load_json(UNITS_CFG, {})
    settings = load_json(SETTINGS_JSON, {"decimal_separator": ".", "llm": {"provider": "openai"}})

    llm_root = settings.get("llm", {})
    provider = (llm_root.get("provider") or "openai").lower()
    temperature = llm_root.get("temperature", 0)

    if provider == "openai":
        openai_conf = llm_root.get("openai", {})
        api_key = openai_conf.get("api_key") or os.getenv("OPENAI_API_KEY") or ""
        model   = openai_conf.get("model", "gpt-4.1-mini")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY manquante (settings.llm.openai.api_key vide et env non défini)")
        llm = {"provider": "openai", "api_key": api_key, "model": model, "temperature": temperature}

    elif provider == "anthropic":
        ant_conf = llm_root.get("anthropic", {})
        api_key = ant_conf.get("api_key") or os.getenv("ANTHROPIC_API_KEY") or ""
        model   = ant_conf.get("model", "claude-3-5-sonnet-latest")
        max_tok = ant_conf.get("max_tokens", 4000)
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY manquante (settings.llm.anthropic.api_key vide et env non défini)")
        llm = {"provider": "anthropic", "api_key": api_key, "model": model, "max_tokens": max_tok, "temperature": temperature}

    else:
        raise ValueError(f"Provider LLM inconnu: {provider}")

    return prompts, units, settings, llm

# ---------------------- PDF → BLOCS ----------------------------------
def chunk_pdf_to_blocks(pdf_path: Path) -> list[dict]:
    blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        for pi, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                content = text[:MAX_BLOCK_CHARS]
                blocks.append({"pdf": pdf_path.name, "page": pi, "section": "texte", "content": content})
            # tables (si dispo)
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for ti, table in enumerate(tables, start=1):
                tsv = "\n".join(["\t".join([(c or "").strip() for c in row]) for row in table])
                tsv = tsv[:MAX_BLOCK_CHARS]
                blocks.append({"pdf": pdf_path.name, "page": pi, "section": f"table_{ti}", "content": tsv})
    return blocks

# ---------------------- LLM CALLS ------------------------------------
def call_llm(system_prompt: str, user_prompt: str, llm_conf: dict) -> list[dict]:
    if llm_conf["provider"] == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=llm_conf["api_key"])
        resp = client.chat.completions.create(
            model=llm_conf["model"],
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
            temperature=llm_conf.get("temperature", 0)
        )
        txt = resp.choices[0].message.content.strip()
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=llm_conf["api_key"])
        msg = client.messages.create(
            model=llm_conf["model"],
            max_tokens=llm_conf.get("max_tokens", 4000),
            system=system_prompt,
            messages=[{"role":"user","content":user_prompt}],
            temperature=llm_conf.get("temperature", 0)
        )
        txt = "".join(getattr(c, "text", "") for c in msg.content)

    s, e = txt.find("["), txt.rfind("]")
    if s == -1 or e == -1:
        return []
    try:
        return json.loads(txt[s:e+1])
    except Exception:
        return []

def llm_extract_block(block: dict, prompts: dict, units_target: dict, decimal_sep: str, llm_conf: dict) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(json.dumps(block, ensure_ascii=False).encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{h}.json"
    if cache_file.exists():
        rows = json.loads(cache_file.read_text(encoding="utf-8"))
        if DEBUG:
            dump("debug_cache_hit.txt", f"HIT {cache_file.name} -> {len(rows)} objets")
        return rows

    # mettre en forme le prompt utilisateur
    try:
        user_prompt = prompts["user_template"].format(
            DECIMAL_SEPARATOR=decimal_sep,
            UNIT_TARGETS_JSON=json.dumps(units_target, ensure_ascii=False, indent=2),
            PDF_NAME=block["pdf"],
            PAGE_N=block["page"],
            SECTION_ID=block["section"],
            BLOCK_TEXT=block["content"]
        )
    except KeyError as e:
        dump("debug_format_error.txt", f"Clé manquante dans user_template: {e}\n\nTEMPLATE:\n{prompts['user_template']}")
        raise

    rows = call_llm(prompts["system"], user_prompt, llm_conf)

    if DEBUG:
        dump("debug_last_prompt.txt", user_prompt)
        dump("debug_last_llm.json", rows)

    cache_file.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows

# ---------------------- MAIN -----------------------------------------
def run_extract(pdf_dir: Path = PDF_DIR, out_dir: Path = OUT_DIR, also_write_csv: bool = False):
    prompts, units_cfg, settings, llm_conf = load_prompts_and_settings()
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_json(INDEX_F, {})
    pdfs = sorted(p for p in pdf_dir.glob("*.pdf"))
    to_process = [(p, file_sig(p)) for p in pdfs if index.get(p.name) != file_sig(p)]
    log(f"[INFO] PDFs à traiter: {len(to_process)}")
    if not to_process:
        log("[INFO] Aucun nouveau PDF à traiter. (Pense à supprimer out/ingest_index.json pour forcer)")
        return

    all_out_rows: List[dict] = []

    for p, sig in to_process:
        log(f"[PDF] {p.name}")
        blocks = chunk_pdf_to_blocks(p)
        log(f"[INFO] {p.name}: {len(blocks)} blocs")
        if DEBUG and blocks:
            sample = f"PDF={p.name} PAGE={blocks[0]['page']} SECTION={blocks[0]['section']}\n\n" + \
                     textwrap.shorten(blocks[0]['content'], width=2000, placeholder=" ...")
            dump("debug_first_block.txt", sample)

        rows_raw = []
        for b in blocks:
            try:
                rows_raw.extend(llm_extract_block(
                    b, prompts, units_cfg.get("units_target", {}),
                    settings.get("decimal_separator", "."), llm_conf
                ))
            except Exception as e:
                # log console
                log(f"[WARN] LLM bloc échoué {p.name} p{b['page']} {b['section']} : {e}")
                # dump stack dans un fichier
                dump("debug_exception_trace.txt", traceback.format_exc())

        log(f"[INFO] bruts: {len(rows_raw)}")
        validated = []
        ok = ko = 0
        errlog = LOG_DIR / "debug_validation_errors.log"

        for obj in rows_raw:
            try:
                validated.append(BilanResult(**obj))
                ok += 1
            except Exception as e:
                ko += 1
                if DEBUG:
                    with errlog.open("a", encoding="utf-8") as f:
                        f.write(f"OBJ={json.dumps(obj, ensure_ascii=False)}\nERR={repr(e)}\n\n")

        log(f"[INFO] validés: {ok} rejetés: {ko}")

        final = normalize_and_dedup(validated, units_cfg, settings)
        dict_rows = [r.model_dump() for r in final]
        append_jsonl(FACTS_JL, dict_rows)
        all_out_rows.extend(dict_rows)

        index[p.name] = sig
        save_json(INDEX_F, index)

    if also_write_csv and all_out_rows:
        write_csv(FACTS_CSV, all_out_rows)
        log(f"[OK] Écrit: {FACTS_JL} et {FACTS_CSV}")
    else:
        log(f"[OK] Écrit: {FACTS_JL} (JSONL canonique)")

if __name__ == "__main__":
    # Purge rapide (manuelle, si besoin) :
    #   Remove-Item out\ingest_index.json
    #   Remove-Item out\.cache\* -Force
    run_extract(also_write_csv=True)
