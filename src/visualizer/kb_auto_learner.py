"""
kb_auto_learner.py  —  Auto-Learning Knowledge Base Appender
=============================================================

When route_ui_action() returns None (no KB match), this module:
  1. Uses Qwen CoT (via existing qwen_client.py, port 8001) to reason about intent
  2. Validates the returned btnId against HUB_BUTTONS registry
  3. Generates a valid ## action: markdown block
  4. Appends it to:
       - nas_storage_app/.qwen/attn_action_kb.md  OR  vina_action_kb.md  (shadow)
       - ui_action_kb/attn_visualization.md        OR  vina_visualization.md  (main)
  5. Hot-injects into the live ElionUIRouter FAISS index (no restart needed)
  6. Returns a ui_action dict so the CURRENT request also highlights

CoT log location
─────────────────
  All auto-learn activity is logged via Python logging to the Flask console.
  Look for lines starting with [AutoLearn].

  To write to a dedicated file, add to app.py before app.run():
      import logging
      fh = logging.FileHandler('nas_storage_app/.qwen/autolearn.log')
      fh.setLevel(logging.DEBUG)
      logging.getLogger().addHandler(fh)

Usage in merged_routes.py (already wired — do not edit)
─────────────────────────────────────────────────────────
  from kb_auto_learner import maybe_learn_and_route
  ui_action = maybe_learn_and_route(message, tool_hint='attn')
"""

import os
import re
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = "."

# Shadow KB: auto-generated, human-auditable
_QWEN_DIR    = os.path.join(_REPO_ROOT, "nas_storage_app", ".qwen")
ATTN_KB_PATH = os.path.join(_QWEN_DIR, "attn_action_kb.md")
VINA_KB_PATH = os.path.join(_QWEN_DIR, "vina_action_kb.md")

# ── Log files ──────────────────────────────────────────────────────────────────
# cot_main.log   : every single CoT call (main log — use this to inspect reasoning)
# autolearn.log  : SUBSET — only entries where the model detected a NEW feature gap
#                  (i.e. matched=False or a genuinely new action was learned)
COT_MAIN_LOG_PATH    = os.path.join(_QWEN_DIR, "cot_main.log")
AUTOLEARN_LOG_PATH   = os.path.join(_QWEN_DIR, "autolearn.log")
# Keep old name as alias so any external code that imports COT_LOG_PATH still works
COT_LOG_PATH         = AUTOLEARN_LOG_PATH

# Main KB: picked up by ElionUIRouter on next restart / hot-inject
_KB_DIR   = os.path.join(_REPO_ROOT, "ui_action_kb")
ATTN_MAIN = os.path.join(_KB_DIR, "attn_visualization.md")
VINA_MAIN = os.path.join(_KB_DIR, "vina_visualization.md")


# ── Hub button registry ───────────────────────────────────────────────────────
# Ground truth of every highlightable element in hub.html.
# Add new buttons here when you add new visualizers — no other change needed.
HUB_BUTTONS: dict[str, str] = {
    # Landing-page nav
    "adjWeightBtn":     "🧠 ChemBERT visualizer button",
    "vinaBtn":          "🔬 Vina Docking button",
    "askElionBtn":      "🚀 Ask Elion button",
    # ChemBERT modal
    "smilesInput":      "SMILES input field",
    "runSingle":        "Visualize button",
    "runCompare":       "Compare button",
    "runFinetune":      "Fine-Tune button",
    "runLoadModel":     "Load Model button",
    # Vina modal
    "vinaLigandPath":   "Ligand Path input field",
    "vinaReceptorPath": "Receptor Path input field",
    "vinaConfigPath":   "Config file path input field",
    "vinaDockBtn":      "Dock button",
}

# Which tool owns each button (determines which KB file to write)
_BUTTON_TOOL: dict[str, str] = {
    "adjWeightBtn":     "attn",
    "smilesInput":      "attn",
    "runSingle":        "attn",
    "runCompare":       "attn",
    "runFinetune":      "attn",
    "runLoadModel":     "attn",
    "vinaBtn":          "vina",
    "vinaLigandPath":   "vina",
    "vinaReceptorPath": "vina",
    "vinaConfigPath":   "vina",
    "vinaDockBtn":      "vina",
    "askElionBtn":      "attn",
}


# =============================================================================
# ── CoT prompt ────────────────────────────────────────────────────────────────
# =============================================================================

def _build_cot_prompt(user_message: str, tool_hint: str | None) -> str:
    buttons_list = "\n".join(
        f"  btnId={bid!r:25s}  label={label!r}"
        for bid, label in HUB_BUTTONS.items()
    )
    tool_ctx = (f"The user is currently in the '{tool_hint}' tool context."
                if tool_hint else "Tool context is unknown.")

    return (
        f"<|im_start|>system\n"
        f"You are Elion's UI knowledge-base generator. "
        f"A user sent a message that did not match any existing UI action. "
        f"Decide which button they mean and produce a new KB entry.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{tool_ctx}\n\n"
        f"Available buttons (ONLY these btnIds are valid):\n{buttons_list}\n\n"
        f"User message: \"{user_message}\"\n\n"
        f"RULE — set matched=true ONLY IF the message explicitly:\n"
        f"  • Names a specific button, tool, or panel (e.g. 'Ask Elion', 'Vina Docking', 'ChemBERT')\n"
        f"  • Asks to highlight, flash, open, or click something specific\n"
        f"  • Asks HOW to do something that requires clicking a specific named button\n"
        f"Set matched=false for: greetings (hi, hello), generic questions,\n"
        f"  pure science Q&A, vague requests with no button/tool name mentioned.\n\n"
        f"Think step-by-step inside <think> tags, then output ONLY valid JSON "
        f"inside <output> tags. No markdown fences.\n\n"
        f"<think>\n"
        f"1. Does the message reference any button, tool name, or UI element?\n"
        f"2. Which btnId best matches? (must be from the list above)\n"
        f"3. 8-10 diverse trigger phrases users might say for the same intent\n"
        f"4. Tool: attn or vina, based on the btnId\n"
        f"5. One-sentence guided response (use {{btn}} as button name placeholder)\n"
        f"</think>\n\n"
        f"<output>\n"
        f"{{\n"
        f'  "matched": true,\n'
        f'  "btnId": "<exact btnId from list or null>",\n'
        f'  "action": "<snake_case_action_name>",\n'
        f'  "tool": "<attn or vina>",\n'
        f'  "triggers": ["phrase1", "phrase2", "phrase3", "phrase4",\n'
        f'               "phrase5", "phrase6", "phrase7", "phrase8"],\n'
        f'  "response": "<one sentence, use {{btn}} placeholder>"\n'
        f"}}\n"
        f"</output>\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# =============================================================================
# ── Qwen CoT call — uses qwen_client.py (port 8001, same as routes.py) ────────
# =============================================================================

def _call_qwen_cot(prompt: str) -> dict | None:
    """
    Call Qwen via the existing qwen_client infrastructure (port 8001).
    Returns parsed JSON from <output> block, or None on any failure.
    """
    try:
        # Import from the package — works whether called from visualizer/ or routes.py
        try:
            from nas_storage_app.qwen_client import qwen_compat, QwenParams
        except ImportError:
            from qwen_client import qwen_compat, QwenParams

        cot_params = QwenParams(temperature=0.2, max_tokens=600, top_p=0.9)
        outputs = qwen_compat(prompt, cot_params)

        if not outputs:
            logger.warning("[AutoLearn] qwen_compat returned empty list")
            return None

        raw = outputs[0].outputs[0].text.strip()
        logger.debug("[AutoLearn] Qwen CoT raw:\n%s", raw)

        # Extract <output>…</output>
        m = re.search(r"<output>(.*?)</output>", raw, re.DOTALL)
        if not m:
            logger.warning("[AutoLearn] No <output> block in Qwen response")
            _write_cot_log(prompt, raw, error="no <output> block")
            return None

        parsed = json.loads(m.group(1).strip())
        _write_cot_log(prompt, raw, parsed=parsed)
        return parsed

    except Exception as exc:
        logger.warning("[AutoLearn] Qwen CoT error: %s", exc)
        _write_cot_log(prompt, "", error=str(exc))
        return None


def _write_cot_log(prompt: str, raw: str,
                   parsed: dict | None = None,
                   error: str | None = None) -> None:
    """
    Write CoT record to the two-tier log system:

      cot_main.log   — receives EVERY call (full prompt tail + raw output + parse)
      autolearn.log  — receives ONLY entries that represent a feature gap:
                         • matched=False  (model couldn't map to any known button)
                         • a new action was successfully learned (matched=True, new triggers)
                         • an error occurred during CoT
                       These are the entries worth reviewing to improve the KB.
    """
    try:
        os.makedirs(_QWEN_DIR, exist_ok=True)
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 60

        # ── Build the shared record body ──────────────────────────────────────
        body_lines = [
            f"\n{sep}",
            f"[{ts}]",
            f"PROMPT (last 400 chars): ...{prompt[-400:]}",
            f"RAW OUTPUT:\n{raw[:800]}",
        ]
        if parsed:
            body_lines.append(f"PARSED: {json.dumps(parsed, indent=2)}")
        if error:
            body_lines.append(f"ERROR: {error}")
        body = "\n".join(body_lines) + "\n"

        # ── Always write to cot_main.log ──────────────────────────────────────
        with open(COT_MAIN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(body)

        # ── Write to autolearn.log only when it is a feature-gap signal ───────
        # Criteria for a feature-gap entry:
        #   1. Hard error during CoT
        #   2. Model returned matched=False (no known button found)
        #   3. Model returned matched=True AND generated new triggers
        #      (this = a genuinely new intent was learned → worth reviewing)
        is_error        = bool(error)
        is_unmatched    = parsed is not None and not parsed.get("matched", True)
        is_new_learning = (parsed is not None
                           and parsed.get("matched")
                           and bool(parsed.get("triggers")))

        if is_error or is_unmatched or is_new_learning:
            label = ("ERROR"      if is_error        else
                     "UNMATCHED"  if is_unmatched     else
                     "NEW_LEARN")
            with open(AUTOLEARN_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"\n{sep}\n")
                f.write(f"[{ts}]  ← {label}\n")
                f.write(body.lstrip("\n"))

    except Exception:
        pass   # log failures are non-fatal


# =============================================================================
# ── Markdown block generator ──────────────────────────────────────────────────
# =============================================================================

def _generate_md_block(parsed: dict, user_message: str) -> str:
    triggers = list(parsed.get("triggers", []))
    # Guarantee the exact user message is always trigger #1
    if user_message.strip().lower() not in [t.lower() for t in triggers]:
        triggers.insert(0, user_message.strip())

    trigger_lines = "\n".join(f"- {t}" for t in triggers)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    return (
        f"\n"
        f"## action: {parsed['action']}\n"
        f"**btnId:** {parsed['btnId']}\n"
        f"**auto-learned:** {ts}\n"
        f"**triggers:**\n"
        f"{trigger_lines}\n"
        f"**response:** {parsed.get('response', 'Click the **{{btn}}** button.')}\n"
    )


# =============================================================================
# ── KB file writer ────────────────────────────────────────────────────────────
# =============================================================================

def _resolve_kb_paths(tool: str) -> tuple[str, str]:
    return (VINA_KB_PATH, VINA_MAIN) if tool == "vina" else (ATTN_KB_PATH, ATTN_MAIN)


def _already_in_file(path: str, btn_id: str, action: str) -> bool:
    if not os.path.exists(path):
        return False
    content = open(path, encoding="utf-8").read()
    return (f"**btnId:** {btn_id}" in content or
            f"## action: {action}" in content)


def _append_to_kb(path: str, md_block: str, tool: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            tool_tag = f"{tool}_visualization"
            header = (
                f"# Tool: {tool_tag}\n"
                f"# Auto-generated by kb_auto_learner.py\n"
                f"# Review periodically and move good entries to ui_action_kb/\n\n"
            )
            open(path, "w", encoding="utf-8").write(header)

        with open(path, "a", encoding="utf-8") as f:
            f.write(md_block)
        logger.info("[AutoLearn] Appended to %s", path)
        return True
    except Exception as exc:
        logger.error("[AutoLearn] Write failed %s: %s", path, exc)
        return False


# =============================================================================
# ── Hot-inject into live ElionUIRouter ───────────────────────────────────────
# =============================================================================

def _hot_inject(btn_id: str, action: str, response: str,
                triggers: list[str], tool: str) -> None:
    """
    Directly adds new records + FAISS vectors to the running router instance.
    The CURRENT request gets the highlight; no Flask restart needed.
    """
    try:
        from elion_ui_router import ElionUIRouter
        router = ElionUIRouter.get()

        tool_name = f"{tool}_visualization"
        new_records = [
            {"tool": tool_name, "action": action, "btnId": btn_id,
             "trigger": t, "response": response, "source": "auto-learned"}
            for t in triggers
        ]
        router.records.extend(new_records)

        if router.encoder and router.index is not None:
            import numpy as np
            texts = [r["trigger"] for r in new_records]
            embs  = router.encoder.encode(texts, show_progress_bar=False)
            embs  = np.array(embs, dtype="float32")
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embs /= norms
            router.index.add(embs)
            logger.info("[AutoLearn] Hot-injected %d FAISS vectors", len(new_records))
        else:
            logger.info("[AutoLearn] Hot-injected %d keyword records (no FAISS)", len(new_records))

    except Exception as exc:
        logger.warning("[AutoLearn] Hot-inject error (works after restart): %s", exc)


# =============================================================================
# ── Public API ────────────────────────────────────────────────────────────────
# =============================================================================

def maybe_learn_and_route(user_message: str,
                          tool_hint: str | None = None,
                          extra_buttons: dict | None = None) -> dict | None:
    """
    Call when route_ui_action() returns None.

    Returns a ui_action dict if Qwen matched a button, else None.
    Side-effects: writes to KB files + hot-injects into live router.
    """
    logger.info("[AutoLearn] Triggered for: %r  hint=%s", user_message, tool_hint)

    if extra_buttons:
        HUB_BUTTONS.update(extra_buttons)

    prompt = _build_cot_prompt(user_message, tool_hint)
    parsed = _call_qwen_cot(prompt)

    if not parsed or not parsed.get("matched"):
        logger.info("[AutoLearn] No match — Qwen says this isn't a button query")
        return None

    btn_id  = parsed.get("btnId")
    action  = (parsed.get("action") or "").strip()
    tool    = parsed.get("tool") or _BUTTON_TOOL.get(btn_id or "", "attn")
    triggers = list(parsed.get("triggers", []))
    response = parsed.get("response", "Click the **{btn}** button.")

    # ── Hallucination guard ───────────────────────────────────────────────────
    if not btn_id or btn_id not in HUB_BUTTONS:
        logger.warning("[AutoLearn] Invalid btnId %r — skipping", btn_id)
        return None
    if not action or not re.match(r'^[a-z][a-z0-9_]+$', action):
        logger.warning("[AutoLearn] Invalid action name %r — skipping", action)
        return None

    # Ensure exact query is in triggers
    if user_message.strip() not in triggers:
        triggers.insert(0, user_message.strip())

    # ── Generate MD & write ───────────────────────────────────────────────────
    md_block   = _generate_md_block(parsed, user_message)
    shadow, main = _resolve_kb_paths(tool)

    if not _already_in_file(shadow, btn_id, action):
        _append_to_kb(shadow, md_block, tool)
    if not _already_in_file(main, btn_id, action):
        _append_to_kb(main, md_block, tool)

    # ── Hot-inject into live FAISS ────────────────────────────────────────────
    _hot_inject(btn_id, action, response, triggers, tool)

    label = HUB_BUTTONS.get(btn_id, btn_id)
    logger.info("[AutoLearn] ✅ Learned — action=%r btnId=%r", action, btn_id)

    return {
        "type":       "ui_action",
        "action":     action,
        "btnId":      btn_id,
        "confidence": "learned",
        "reason":     f"auto-learned: {user_message!r}",
        "response":   response.replace("{btn}", label),
    }


def register_buttons(new_buttons: dict[str, str],
                     tool_hint: str = "attn") -> None:
    """Register buttons from a new visualizer so auto-learning can target them."""
    HUB_BUTTONS.update(new_buttons)
    for bid in new_buttons:
        _BUTTON_TOOL[bid] = tool_hint
    logger.info("[AutoLearn] Registered %d buttons for '%s'",
                len(new_buttons), tool_hint)


# =============================================================================
# ── Self-test ─────────────────────────────────────────────────────────────────
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)s %(name)s: %(message)s")
    print("Registered buttons:")
    for bid, label in HUB_BUTTONS.items():
        print(f"  {bid:24s} → {label}")
    print(f"\nCoT log → {COT_LOG_PATH}")
    print(f"Shadow KB → {ATTN_KB_PATH}")
    print(f"           {VINA_KB_PATH}")