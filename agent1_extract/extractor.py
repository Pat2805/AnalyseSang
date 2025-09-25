# agent1_extract/extractor.py
from __future__ import annotations
import os, json, csv, hashlib, time
from pathlib import Path
from typing import Dict, List, Iterable
import pdfplumber
import yaml

from agent1_extract.models import BilanResult
from agent1_extract.normalize import normalize_and_dedup

PDF_DIR   = Path("data/pdf")
OUT_DIR   = Path("out")
CACHE_DIR = OUT_DIR / ".cache"
INDEX_F   = OUT_DIR / "ingest_index.json"
FACTS_JL  = OUT_DIR / "facts.jsonl"
FACTS_CSV = OUT_DIR / "facts.csv"

PROMPTS_YAML = Path("agent1_extract/prompts.yaml")
UNITS_CFG    = Path("agent1_extract/units_config.json")
SETTINGS_JSON= Path("agent1_extract/settings.json")

# ---------- helpers ----------
def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default

def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def append_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def file_sig(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"

# ---------- prompts & settings ----------
def load_prompts_and_settings():
    prompts  = load_yaml(PROMPTS_YAML)
    units    = load_json(UNITS_CFG, {})
    settings = load_json(SETTINGS_JSON, {"decimal_separator": ".", "llm": {"provider": "openai"}})

    # Provider & keys with fallback
    prov = (settings.get("llm", {}).get("provider") or "openai").lower()
    temperature = settings.get("llm", {}).get("temperature", 0)

    if prov == "openai":
        openai_conf = settings["llm"].get("openai", {})
        api_key = openai_conf.get("api_key") or os.getenv("OPENAI_API_KEY") or ""
        model   = openai_conf.get("model", "gpt-4.1-mini")
        if not api_key:
            raise RuntimeError("OpenAI API key manquante (settings.llm.openai.api_key vide et OPENAI_API_KEY non définie).")
        llm = {"provider": "openai", "api_key": api_key, "model": model, "temperature": temperature}

    elif prov == "anthropic":
        ant_conf = settings["llm"].get("anthropic", {})
        api_key = ant_conf.get("api_key") or os.getenv("ANTHROPIC_API_KEY") or ""
        model   = ant_conf.get("model", "claude-3-5-sonnet-latest")
        max_toks= ant_conf.get("max_tokens", 4000)
        if not api_key:
            raise RuntimeError("Anthropic API key manquante (settings.llm.anthropic.api_key vide et ANTHROPIC_API_KEY non définie).")
        llm = {"provider": "anthropic", "api_key": api_key, "model": model, "max_tokens": max_toks, "temperature": temperature}
    else:
        raise ValueError(f"Provider LLM inconnu: {prov}")

    return prompts, units, settings, llm

# ---------- chunk pdf ----------
def chunk_pdf_to_blocks(pdf_path: Path) -> List[dict]:
    blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        for pi, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                blocks.append({"pdf": pdf_path.name, "page": pi, "section": "texte", "content": text})
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for ti, table in enumerate(tables, start=1):
                tsv = "\n".join(["\t".join([(c or "").strip() for c in row]) for row in table])
                blocks.append({"pdf": pdf_path.name, "page": pi, "section": f"table_{ti}", "content": tsv})
    return blocks

# ---------- LLM calls ----------
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
    return json.loads(txt[s:e+1])

def llm_extract_block(block: dict, prompts: dict, units_target: dict, decimal_sep: str, llm_conf: dict) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(json.dumps(block, ensure_ascii=False).encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{h}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    user_prompt = prompts["user_template"].format(
        DECIMAL_SEPARATOR=decimal_sep,
        UNIT_TARGETS_JSON=json.dumps(units_target, ensure_ascii=False, indent=2),
        PDF_NAME=block["pdf"],
        PAGE_N=block["page"],
        SECTION_ID=block["section"],
        BLOCK_TEXT=block["content"]
    )
    rows = call_llm(prompts["system"], user_prompt, llm_conf)
    cache_file.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows

# ---------- main ----------
def run_extract(pdf_dir: Path = PDF_DIR, out_dir: Path = OUT_DIR, also_write_csv: bool = False):
    prompts, units_cfg, settings, llm_conf = load_prompts_and_settings()
    out_dir.mkdir(parents=True, exist_ok=True)

    index = load_json(INDEX_F, {})
    pdfs = sorted(p for p in pdf_dir.glob("*.pdf"))
    to_process = [(p, file_sig(p)) for p in pdfs if index.get(p.name) != file_sig(p)]
    if not to_process:
        print("Aucun nouveau PDF à traiter.")
        return

    all_out_rows = []

    for p, sig in to_process:
        print(f"[PDF] {p.name}")
        blocks = chunk_pdf_to_blocks(p)
        rows_raw = []
        for b in blocks:
            try:
                rows_raw.extend(llm_extract_block(b, prompts, units_cfg.get("units_target", {}), settings.get("decimal_separator", "."), llm_conf))
            except Exception:
                # en cas d'échec sur un bloc, on continue, mais on pourrait logguer
                continue

        # validation pydantic
        validated = []
        for obj in rows_raw:
            try:
                validated.append(BilanResult(**obj))
            except Exception:
                continue

        final = normalize_and_dedup(validated, units_cfg, settings)
        dict_rows = [r.model_dump() for r in final]
        append_jsonl(FACTS_JL, dict_rows)
        all_out_rows.extend(dict_rows)

        index[p.name] = sig
        save_json(INDEX_F, index)

    if also_write_csv and all_out_rows:
        write_csv(FACTS_CSV, all_out_rows)

if __name__ == "__main__":
    run_extract(also_write_csv=True)


