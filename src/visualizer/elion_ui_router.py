"""
elion_ui_router.py  —  RAG-powered UI Action Router for Elion AGI Platform
===========================================================================

Replaces ALL hardcoded keyword dicts in merged_routes.py.

Architecture
────────────
  ui_action_kb/*.md   ← one file per visualizer tool
       ↓  parsed on startup
  FAISS IndexFlatIP   ← cosine-similarity over normalised MiniLM embeddings
       ↓  query at request time
  route_ui_action()   ← returns {type, action, btnId, confidence, reason, response}

Adding a new visualizer
───────────────────────
  1. Create  ui_action_kb/my_new_tool.md  (copy TEMPLATE.md, fill sections)
  2. Restart Flask  →  index auto-rebuilds (mtime check), no code changes

KB file format
──────────────
  # Tool: tool_name_matching_tool_hint
  ## action: some_action
  **btnId:** htmlElementId
  **triggers:**
  - natural language phrase users might say
  - another phrase
  **response:** Guided text for mini-chat. Use {btn} as placeholder.

Paths
─────
  KB dir  : <this_file's_dir>/ui_action_kb/
  Index   : <this_file's_dir>/ui_action_router.index
  Encoder : ../LLM/all-MiniLM-L6-v2  (or absolute fallback)
"""

import os
import re
import json
import glob
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Optional FAISS + SentenceTransformer ──────────────────────────────────────
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False
    logger.warning("[UIRouter] faiss/sentence_transformers unavailable — keyword fallback active")

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
KB_DIR       = os.path.join(_BASE, "ui_action_kb")
INDEX_PATH   = os.path.join(_BASE, "ui_action_router.index")
DOCS_PATH    = INDEX_PATH + ".json"

# Encoder: try relative path first (same layout as ElionKnowledgeBase)
ENCODER_PATH = os.path.join(_BASE, "..", "LLM", "all-MiniLM-L6-v2")
if not os.path.exists(ENCODER_PATH):
    ENCODER_PATH = "../../../Elion-AGI-Ecosystem/LLM/all-MiniLM-L6-v2"

CONFIDENCE_THRESHOLD = 0.42   # cosine similarity floor (0–1, empirically tuned)


# =============================================================================
# ── Markdown KB Parser ────────────────────────────────────────────────────────
# =============================================================================

def _parse_kb_file(path: str) -> list[dict]:
    """
    Parse a single ui_action_kb/*.md file into a flat list of trigger records.

    Each record:  {tool, action, btnId, trigger, response, source}
    One record per trigger phrase (each becomes one FAISS vector).
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    # ── Tool name from first heading ──────────────────────────────────────────
    m = re.search(r'^#\s+Tool:\s*(\S+)', raw, re.MULTILINE)
    tool = m.group(1) if m else os.path.splitext(os.path.basename(path))[0]
    source = os.path.basename(path)

    records = []
    # Split on ## action: boundaries
    sections = re.split(r'^##\s+action:\s*', raw, flags=re.MULTILINE)

    for section in sections[1:]:          # skip file header
        lines = section.strip().splitlines()
        if not lines:
            continue
        action = lines[0].strip()

        # btnId
        bm = re.search(r'\*\*btnId:\*\*\s*(\S+)', section)
        btn_id = bm.group(1) if bm else None

        # trigger phrases  (lines starting with "- " inside **triggers:** block)
        triggers = []
        in_triggers = False
        for line in lines[1:]:
            if '**triggers:**' in line.lower():
                in_triggers = True
                continue
            if in_triggers:
                if line.startswith('**') or line.startswith('##'):
                    in_triggers = False
                elif line.strip().startswith('-'):
                    t = line.strip().lstrip('-').strip()
                    if t:
                        triggers.append(t)

        # response template
        rm = re.search(r'\*\*response:\*\*\s*(.+)', section)
        response = rm.group(1).strip() if rm else ""

        base = dict(tool=tool, action=action, btnId=btn_id,
                    response=response, source=source)

        # One record per trigger phrase
        for trigger in triggers:
            records.append({**base, "trigger": trigger})

        # Also index "tool action" string so exact tool+action queries match
        synthetic = f"{tool} {action}".replace("_", " ")
        records.append({**base, "trigger": synthetic})

    return records


# =============================================================================
# ── ElionUIRouter ─────────────────────────────────────────────────────────────
# =============================================================================

class ElionUIRouter:
    """
    Singleton RAG router.  Get the shared instance with:  ElionUIRouter.get()
    """
    _instance: "ElionUIRouter | None" = None

    @classmethod
    def get(cls) -> "ElionUIRouter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── init ──────────────────────────────────────────────────────────────────
    def __init__(self):
        self.records: list[dict] = []
        self.index = None
        self.encoder = None
        os.makedirs(KB_DIR, exist_ok=True)
        self._load_encoder()
        self._build_or_load_index()

    # ── encoder ───────────────────────────────────────────────────────────────
    def _load_encoder(self):
        if not _FAISS_OK:
            return
        try:
            if os.path.exists(ENCODER_PATH):
                logger.info("[UIRouter] Loading encoder from %s", ENCODER_PATH)
            else:
                logger.warning("[UIRouter] Encoder path not found: %s", ENCODER_PATH)
            self.encoder = SentenceTransformer(ENCODER_PATH, device="cpu")
        except Exception as exc:
            logger.warning("[UIRouter] Encoder load failed: %s", exc)
            self.encoder = None

    # ── index lifecycle ───────────────────────────────────────────────────────
    def _kb_mtime(self) -> float:
        mtimes = [os.path.getmtime(f)
                  for f in glob.glob(os.path.join(KB_DIR, "*.md"))]
        return max(mtimes) if mtimes else 0.0

    def _index_mtime(self) -> float:
        return os.path.getmtime(INDEX_PATH) if os.path.exists(INDEX_PATH) else 0.0

    def _build_or_load_index(self):
        kb_files = glob.glob(os.path.join(KB_DIR, "*.md"))
        if not kb_files:
            logger.warning("[UIRouter] No KB files found in %s", KB_DIR)
            return

        need_rebuild = (
            self._kb_mtime() > self._index_mtime()
            or not os.path.exists(INDEX_PATH)
            or not os.path.exists(DOCS_PATH)
        )

        if not need_rebuild:
            self._load_index()
            if self.records:
                logger.info("[UIRouter] Loaded cached index (%d records)", len(self.records))
                return

        # ── Parse all KB files ────────────────────────────────────────────────
        all_records: list[dict] = []
        for path in sorted(kb_files):
            if os.path.basename(path) == "TEMPLATE.md":
                continue           # skip the template
            try:
                recs = _parse_kb_file(path)
                all_records.extend(recs)
                logger.info("[UIRouter] Parsed %s → %d records",
                            os.path.basename(path), len(recs))
            except Exception as exc:
                logger.warning("[UIRouter] Failed to parse %s: %s", path, exc)

        self.records = all_records
        if not all_records:
            logger.warning("[UIRouter] No records parsed — check KB files")
            return

        # ── Build FAISS index ─────────────────────────────────────────────────
        if _FAISS_OK and self.encoder:
            texts = [r["trigger"] for r in all_records]
            embs  = self.encoder.encode(texts, show_progress_bar=False,
                                        batch_size=64)
            embs  = np.array(embs, dtype="float32")
            # Normalise → cosine similarity via IndexFlatIP
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embs /= norms
            self.index = faiss.IndexFlatIP(embs.shape[1])
            self.index.add(embs)
            self._save_index()
            logger.info("[UIRouter] Built FAISS index — %d vectors", len(all_records))
        else:
            logger.info("[UIRouter] Keyword-only mode — %d records", len(all_records))

    def _save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, INDEX_PATH)
        with open(DOCS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def _load_index(self):
        try:
            if _FAISS_OK and os.path.exists(INDEX_PATH):
                self.index = faiss.read_index(INDEX_PATH)
            with open(DOCS_PATH, encoding="utf-8") as f:
                self.records = json.load(f)
        except Exception as exc:
            logger.warning("[UIRouter] Cache load failed: %s", exc)
            self.records = []
            self.index   = None

    # ── Public query ──────────────────────────────────────────────────────────
    def route(self, query: str, tool_hint: str | None = None,
              top_k: int = 5) -> dict | None:
        """
        Match query → best UI action.

        Args:
            query:     the user's chat message
            tool_hint: 'attn' | 'vina' | None
            top_k:     candidates to consider before filtering

        Returns dict with keys:
            tool, action, btnId, response, confidence, reason
        or None if nothing scored above threshold.
        """
        if not self.records:
            return None

        # ── FAISS semantic search ─────────────────────────────────────────────
        if _FAISS_OK and self.encoder and self.index is not None:
            q_vec  = self.encoder.encode([query], show_progress_bar=False)
            q_vec  = np.array(q_vec, dtype="float32")
            norm   = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec /= norm

            k = min(top_k * 6, len(self.records))
            scores, idxs = self.index.search(q_vec, k)

            candidates = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx == -1:
                    continue
                rec = self.records[idx]
                if tool_hint and not self._tool_matches(rec["tool"], tool_hint):
                    continue
                candidates.append((float(score), rec))

            if candidates:
                best_score, best_rec = candidates[0]
                if best_score >= CONFIDENCE_THRESHOLD:
                    return {
                        "tool":       best_rec["tool"],
                        "action":     best_rec["action"],
                        "btnId":      best_rec["btnId"],
                        "response":   best_rec["response"],
                        "confidence": "high" if best_score > 0.70 else "medium",
                        "reason":     (f"RAG: '{best_rec['trigger']}' "
                                       f"cosine={best_score:.3f}"),
                    }

        # ── Keyword fallback ──────────────────────────────────────────────────
        return self._keyword_route(query, tool_hint)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _tool_matches(tool: str, hint: str) -> bool:
        hint = hint.lower(); tool = tool.lower()
        return (
            hint in tool or
            (hint in ("attn", "attention", "chembert", "chem") and
             any(k in tool for k in ("attn", "chem", "attention"))) or
            (hint == "vina" and "vina" in tool)
        )

    def _keyword_route(self, query: str,
                       tool_hint: str | None) -> dict | None:
        """Substring keyword fallback when FAISS is unavailable."""
        u = query.lower()
        best: tuple[float, dict] | None = None
        for rec in self.records:
            if tool_hint and not self._tool_matches(rec["tool"], tool_hint):
                continue
            words = [w for w in rec["trigger"].lower().split() if len(w) > 3]
            if not words:
                continue
            hits  = sum(1 for w in words if w in u)
            score = hits / len(words)
            if score > 0.45 and (best is None or score > best[0]):
                best = (score, rec)
        if best:
            score, rec = best
            return {
                "tool":       rec["tool"],
                "action":     rec["action"],
                "btnId":      rec["btnId"],
                "response":   rec["response"],
                "confidence": "high" if score > 0.8 else "medium",
                "reason":     f"keyword: '{rec['trigger']}' score={score:.2f}",
            }
        return None

    def reload(self):
        """Force-rebuild index from KB files (call after adding new tools)."""
        ElionUIRouter._instance = None
        self.records = []
        self.index   = None
        self._build_or_load_index()
        ElionUIRouter._instance = self
        logger.info("[UIRouter] Reloaded — %d records", len(self.records))


# =============================================================================
# ── Drop-in replacement for keyword route functions in merged_routes.py ───────
# =============================================================================

def route_ui_action(user_message: str,
                    tool_hint: str | None = None) -> dict | None:
    """
    Call this instead of _keyword_route_attn() / _keyword_route_action().

    Returns the SSE-ready dict the existing handlers already understand:
        {type, action, btnId, confidence, reason, response}
    or None if no match.

    Usage in routes.py:
        from elion_ui_router import route_ui_action
        ...
        result = route_ui_action(user_msg, tool_hint='attn')
        if result:
            yield f"data: {json.dumps(result)}\\n\\n"
    """
    hit = ElionUIRouter.get().route(user_message, tool_hint=tool_hint)
    if hit is None:
        return None
    return {
        "type":       "ui_action",
        "action":     hit["action"],
        "btnId":      hit["btnId"],
        "confidence": hit["confidence"],
        "reason":     hit["reason"],
        "response":   hit.get("response", ""),
    }


# =============================================================================
# ── Quick self-test  (python elion_ui_router.py) ──────────────────────────────
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    router = ElionUIRouter.get()
    print(f"\nIndex ready: {len(router.records)} records\n")

    tests = [
        ("what is ChemBERT",         "attn"),
        ("show me the 3D view",      "attn"),
        ("then what",                "attn"),
        ("compare two molecules",    "attn"),
        ("open vina docking",        "vina"),
        ("run the docking now",      "vina"),
        ("now what",                 "vina"),
        ("enter receptor path",      "vina"),
        ("I want to dock a ligand",  None),
        ("launch the visualizer",    None),
        ("random unrelated text xyz", None),
    ]
    for query, hint in tests:
        res = route_ui_action(query, hint)
        if res:
            print(f"  [{hint or ' * '}] '{query}'")
            print(f"         → action={res['action']}  btnId={res['btnId']}"
                  f"  conf={res['confidence']}")
            print(f"           reason: {res['reason']}\n")
        else:
            print(f"  [{hint or ' * '}] '{query}'  → NO MATCH\n")