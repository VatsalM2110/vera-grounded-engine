"""Vera challenge bot: Groq-powered, grounded and stateful message composition."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_VERSION = "1.1.0"
STARTED = time.time()
DB_PATH = Path(os.getenv("VERA_DB_PATH", Path(__file__).with_name("vera.db")))
VALID_SCOPES = {"category", "merchant", "customer", "trigger"}
LOCK = threading.RLock()
LAST_GROQ_ERROR: Optional[str] = None


def groq_enabled() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def groq_generate(system: str, user: str, max_tokens: int = 240) -> Optional[str]:
    """Generate one grounded response, returning None so callers can safely fall back."""
    global LAST_GROQ_ERROR
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    payload = {
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.35,
        "max_completion_tokens": max_tokens,
    }
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("GROQ_TIMEOUT_SECONDS", "8"))) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = result["choices"][0]["message"]["content"].strip()
        LAST_GROQ_ERROR = None
        return clamp(text) if text else None
    except urllib.error.HTTPError as exc:
        LAST_GROQ_ERROR = f"http_{exc.code}"
        return None
    except (urllib.error.URLError, TimeoutError) as exc:
        LAST_GROQ_ERROR = type(exc).__name__
        return None
    except (ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
        LAST_GROQ_ERROR = type(exc).__name__
        return None


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _db() as conn:
        conn.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS contexts (
          scope TEXT NOT NULL, context_id TEXT NOT NULL, version INTEGER NOT NULL,
          payload TEXT NOT NULL, delivered_at TEXT NOT NULL,
          PRIMARY KEY(scope, context_id));
        CREATE TABLE IF NOT EXISTS conversations (
          conversation_id TEXT PRIMARY KEY, merchant_id TEXT NOT NULL,
          customer_id TEXT, trigger_id TEXT NOT NULL, trigger_kind TEXT NOT NULL,
          last_body TEXT NOT NULL, auto_reply_count INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL DEFAULT 'open', updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS suppressions (
          suppression_key TEXT PRIMARY KEY, sent_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS turns (
          id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
          role TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL);
        """)


init_db()
app = FastAPI(title="Vera Grounded Message Engine", version=APP_VERSION)


class ContextPush(BaseModel):
    scope: str
    context_id: str = Field(min_length=1)
    version: int = Field(ge=0)
    payload: dict[str, Any]
    delivered_at: str


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str = "merchant"
    message: str
    received_at: str
    turn_number: int = 1


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_context(scope: str, context_id: Optional[str]) -> Optional[dict[str, Any]]:
    if not context_id:
        return None
    with _db() as conn:
        row = conn.execute(
            "SELECT payload FROM contexts WHERE scope=? AND context_id=?", (scope, context_id)
        ).fetchone()
    return json.loads(row["payload"]) if row else None


def first_name(merchant: dict[str, Any]) -> str:
    identity = merchant.get("identity", {})
    name = identity.get("owner_name") or identity.get("contact_name") or identity.get("name", "there")
    name = re.sub(r"(?:'s)?\s+(Dental|Dentist|Clinic|Salon|Pharmacy|Restaurant|Cafe|Gym|Fitness).*", "", name, flags=re.I)
    return " ".join(name.split()[:2])


def pct(value: Any) -> str:
    try:
        return f"{abs(float(value)) * 100:.0f}%"
    except (TypeError, ValueError):
        return ""


def active_offer(merchant: dict[str, Any], category: dict[str, Any]) -> Optional[str]:
    for offer in merchant.get("offers", []):
        if offer.get("status") == "active" and offer.get("title"):
            return offer["title"]
    catalog = category.get("offer_catalog", [])
    if catalog:
        return catalog[0].get("title") if isinstance(catalog[0], dict) else str(catalog[0])
    return None


def find_digest(category: dict[str, Any], item_id: Optional[str]) -> Optional[dict[str, Any]]:
    items = category.get("digest", [])
    if item_id:
        for item in items:
            if item.get("id") == item_id:
                return item
    return items[-1] if items else None  # newest injected item wins when no id was supplied


def clamp(text: str, limit: int = 950) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


def customer_message(kind: str, merchant: dict[str, Any], category: dict[str, Any],
                     trigger: dict[str, Any], customer: dict[str, Any]) -> tuple[str, str, str]:
    p = trigger.get("payload", {})
    cname = customer.get("identity", {}).get("name", "there")
    mname = merchant.get("identity", {}).get("name", "our team")
    offer = active_offer(merchant, category)
    slots = p.get("available_slots") or p.get("next_session_options") or []
    slot = slots[0].get("label") if slots and isinstance(slots[0], dict) else None
    if kind == "recall_due":
        due = str(p.get("due_date") or "your next visit").replace("_", " ")
        detail = f" We have {slot} available." if slot else ""
        deal = f" {offer}." if offer else ""
        return (f"Hi {cname}, {mname} here. Your {str(p.get('service_due', 'follow-up')).replace('_', ' ')} is due around {due}.{deal}{detail} Shall I reserve it for you? Reply YES or suggest another time.", "binary", "Due service, real offer and preferred slot create a low-friction booking step")
    if kind in {"customer_lapsed_hard", "customer_lapsed_soft"}:
        days = p.get("days_since_last_visit")
        lapse = f"It has been {days} days" if days else "It has been a while"
        focus = str(p.get("previous_focus", "your routine")).replace("_", " ")
        deal = f" {offer} is available." if offer else ""
        return (f"Hi {cname} 👋 {mname} here. {lapse} since we saw you—no judgment. We can help you ease back into {focus}.{deal} Want me to hold a no-pressure slot? Reply YES.", "binary", "Warm no-shame winback uses known relationship state and an active offer")
    if kind == "chronic_refill_due":
        meds = ", ".join(p.get("molecule_list", []))
        if not meds:
            return (f"Hi {cname}, {mname} here. A routine follow-up is due based on your recent relationship with us. Would you like our team to check the details and confirm the next step? Reply YES.", "binary", "Sparse generated trigger is handled without inventing medicines, dates, or clinical claims")
        when = str(p.get("stock_runs_out_iso", "soon")).split("T")[0]
        delivery = " Delivery to your saved address is available." if p.get("delivery_address_saved") else ""
        return (f"Namaste {cname}, {mname} here. Your {meds or 'regular medicines'} are expected to run out on {when}.{delivery} Reply CONFIRM to prepare the same pack, or tell us if the prescription changed.", "binary", "Precise refill timing and medicine names support a safe confirmation step")
    if kind == "appointment_tomorrow":
        return (f"Hi {cname}, a quick reminder from {mname}: your appointment is tomorrow. Reply CONFIRM to keep it, or CHANGE and we will help find another time.", "binary", "Timely appointment reminder offers one simple confirmation path")
    if kind in {"trial_followup", "wedding_package_followup"}:
        anchor = slot or str(p.get("wedding_date") or p.get("trial_date") or "your next session")
        deal = f" {offer}." if offer else ""
        return (f"Hi {cname}, {mname} here. Following up after your trial—{anchor} is the right next window.{deal} Want me to hold your preferred slot? Reply YES.", "binary", "Relationship continuity and a concrete next window reduce booking effort")
    return (f"Hi {cname}, {mname} here with a quick update relevant to your recent visit. Want the details? Reply YES.", "binary", "Customer-scoped trigger is acknowledged without inventing unsupported facts")


def merchant_message(kind: str, merchant: dict[str, Any], category: dict[str, Any],
                     trigger: dict[str, Any]) -> tuple[str, str, str]:
    p = trigger.get("payload", {})
    name = first_name(merchant)
    ident = merchant.get("identity", {})
    performance = merchant.get("performance", {})
    offer = active_offer(merchant, category)
    if kind in {"research_digest", "regulation_change", "cde_opportunity"}:
        item = find_digest(category, p.get("top_item_id") or p.get("digest_item_id"))
        if item:
            title = item.get("title") or item.get("summary", "New category update")
            source = item.get("source")
            detail = f" ({source})" if source else ""
            prefix = "Action needed" if kind == "regulation_change" else "Worth a look"
            return (f"{name}, {prefix}: {title}{detail}. This is relevant to {ident.get('locality', 'your business')}. Want me to turn it into a short customer message you can review?", "binary", "Latest source-cited category insight is tied to the merchant and converted into an actionable draft")
    if kind in {"perf_dip", "seasonal_perf_dip", "perf_spike"}:
        metric = p.get("metric", "visibility")
        delta = p.get("delta_pct")
        if delta is None:
            delta = performance.get("delta_7d", {}).get(f"{metric}_pct")
        direction = "up" if (delta or 0) > 0 else "down"
        change = pct(delta)
        base = p.get("vs_baseline") or performance.get(metric)
        anchor = f" from a {base} baseline" if base is not None else ""
        if kind == "seasonal_perf_dip" and p.get("is_expected_seasonal"):
            return (f"{name}, your {metric} are down {change} this week{anchor}, but the context marks this as a normal seasonal dip—not a reason to panic. Want me to draft one retention campaign using your current customer base?", "binary", "Reframes an expected seasonal dip and proposes retention instead of unsupported spend")
        action = "repeat the likely winning activity" if direction == "up" else "refresh your listing and strongest offer"
        return (f"{name}, your {metric} are {direction} {change} this week{anchor}. Best next move: {action}. Want me to draft the update using {offer or 'your current service mix'}?", "binary", "Uses the exact performance change and a grounded next step")
    if kind == "supply_alert":
        batches = ", ".join(p.get("affected_batches", []))
        return (f"{name}, urgent stock check: {p.get('molecule', 'a medicine')} batches {batches} were flagged by {p.get('manufacturer', 'the manufacturer')}. Please verify inventory before dispensing. Want me to draft a precise customer replacement note?", "binary", "Safety-first alert repeats exact product and batch facts without adding medical claims")
    if kind == "ipl_match_today":
        match_time = str(p.get("match_time_iso", "")).split("T")[-1][:5]
        offer_text = offer or "your active offer"
        return (f"{name}, {p.get('match', 'the match')} is at {p.get('venue', ident.get('city', 'your city'))} today at {match_time}. Since it is {'a weeknight' if p.get('is_weeknight') else 'the weekend'}, position {offer_text} for at-home orders. Want me to draft one delivery banner?", "binary", "Match timing, day type and current offer are combined without invented demand claims")
    if kind == "review_theme_emerged":
        return (f"{name}, {p.get('occurrences_30d', 'multiple')} recent reviews mention {str(p.get('theme', 'the same issue')).replace('_', ' ')}; the trend is {p.get('trend', 'rising')}. Want a reply template plus a 3-step fix checklist?", "binary", "Repeated review evidence is converted into response and operational action")
    if kind == "renewal_due":
        days = p.get("days_remaining") or merchant.get("subscription", {}).get("days_remaining")
        plan = p.get("plan") or merchant.get("subscription", {}).get("plan", "plan")
        return (f"{name}, your {plan} plan has {days} days remaining. Before you decide, I can summarize the last 30 days: {performance.get('views', 0)} views and {performance.get('calls', 0)} calls. Want the one-page value summary?", "binary", "Renewal reminder earns attention with the merchant's real performance")
    if kind == "active_planning_intent":
        topic = str(p.get("intent_topic", "your plan")).replace("_", " ")
        return (f"{name}, moving straight to execution on {topic}: I’ll structure the offer, pricing tiers and customer copy using your existing catalog. Reply GO and I’ll produce the ready-to-send draft now.", "binary", "Explicit planning intent triggers execution instead of another qualification loop")
    if kind == "milestone_reached":
        current = p.get("value_now")
        target = p.get("milestone_value")
        if current is None:
            return (f"{name}, your account has a fresh milestone signal. Want me to verify the underlying metric and draft a short thank-you post once confirmed?", "binary", "Sparse milestone trigger is acknowledged without mislabeling an unrelated performance metric")
        fact = f"you are at {current}, just short of {target}" if target else f"you reached {current}"
        return (f"{name}, a useful milestone: {fact}. Want me to draft a short thank-you post that turns the moment into fresh reviews?", "binary", "Concrete milestone becomes an easy reputation action")
    if kind == "competitor_opened":
        detail = f"{p.get('competitor_name')} opened {p.get('distance_km')} km away with {p.get('their_offer')}" if p.get("competitor_name") else "a nearby competitor has opened"
        return (f"{name}, {detail}. I would not race to the bottom; lead with {offer or 'your strongest service'} and your existing customer proof. Want a sharper listing headline?", "binary", "Competitive signal prompts differentiated positioning without fabricating competitor facts")
    if kind in {"festival_upcoming", "category_seasonal"}:
        occasion = p.get("festival") or str(p.get("season", "the upcoming season")).replace("_", " ")
        return (f"{name}, {occasion} is the next relevant demand moment for {ident.get('locality', 'your area')}. Want one timely campaign built around {offer or 'your strongest service'}?", "binary", "Seasonal moment is tied to locality and a real offer")
    if kind == "gbp_unverified":
        return (f"{name}, your profile is still unverified; the available path is {str(p.get('verification_path', 'verification')).replace('_', ' ')}. Want the exact 3-step checklist to complete it?", "binary", "Profile-state trigger gets a concrete low-effort next step")
    if kind in {"curious_ask_due", "dormant_with_vera"}:
        return (f"{name}, quick operator question: what service has customers asked for most this week at {ident.get('name', 'your business')}? I’ll turn your answer into one Google post and a reusable pricing reply.", "open_ended", "A low-effort merchant insight unlocks two useful artifacts")
    if kind == "winback_eligible":
        return (f"{name}, since the plan expired {p.get('days_since_expiry', 'several')} days ago, {p.get('lapsed_customers_added_since_expiry', 'more')} customers moved into the lapsed group and performance changed {pct(p.get('perf_dip_pct'))}. Want a no-cost win-back draft before discussing renewal?", "binary", "Leads with measured value and reciprocity rather than a discount pitch")
    return (f"{name}, there is a new {kind.replace('_', ' ')} signal for {ident.get('name', 'your business')}. Want me to turn it into one practical action using your current data?", "binary", "Grounded fallback names only facts supplied by the trigger and merchant context")


def compose(trigger: dict[str, Any], merchant: dict[str, Any], category: dict[str, Any],
            customer: Optional[dict[str, Any]]) -> tuple[str, str, str]:
    kind = trigger.get("kind", "update")
    audience = "customer on the merchant's behalf" if customer else "merchant as Vera"
    facts = {"trigger": trigger, "merchant": merchant, "category": category, "customer": customer}
    ai_message = groq_generate(
        "You are Vera, a concise WhatsApp business assistant for Indian local merchants. "
        "Write exactly one natural message under 700 characters. Use ONLY facts in the supplied JSON; "
        "never invent prices, dates, performance, medical advice, URLs, or capabilities. Make the message "
        "specific, helpful, warm, and end with one clear low-friction question. Return only the message.",
        f"Write to the {audience}. Trigger kind: {kind}. Grounding JSON: {json.dumps(facts, ensure_ascii=False)}",
    )
    if ai_message:
        return ai_message, "binary", f"Groq generated a natural action grounded only in the supplied {kind} context"
    if trigger.get("scope") == "customer" and customer:
        return customer_message(kind, merchant, category, trigger, customer)
    return merchant_message(kind, merchant, category, trigger)


def is_expired(expires_at: Optional[str], now: str) -> bool:
    if not expires_at:
        return False
    try:
        return datetime.fromisoformat(expires_at.replace("Z", "+00:00")) < datetime.fromisoformat(now.replace("Z", "+00:00"))
    except ValueError:
        return False


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Vera Grounded Message Engine", "status": "ok", "version": APP_VERSION}


@app.get("/v1/healthz")
def healthz() -> dict[str, Any]:
    counts = {s: 0 for s in VALID_SCOPES}
    with _db() as conn:
        for row in conn.execute("SELECT scope, COUNT(*) AS n FROM contexts GROUP BY scope"):
            counts[row["scope"]] = row["n"]
    return {"status": "ok", "uptime_seconds": int(time.time() - STARTED), "contexts_loaded": counts}


@app.get("/v1/metadata")
def metadata() -> dict[str, Any]:
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b") if groq_enabled() else "deterministic-rules-v1"
    return {"team_name": os.getenv("TEAM_NAME", "Vera Grounded"), "team_members": [os.getenv("TEAM_MEMBER", "Candidate")], "model": model, "ai_enabled": groq_enabled(), "approach": "Groq-generated grounded messages with deterministic safety fallback, trigger ranking, suppression, and reply state", "contact_email": os.getenv("CONTACT_EMAIL", "candidate@example.com"), "version": APP_VERSION, "submitted_at": os.getenv("SUBMITTED_AT", "2026-09-27T00:00:00Z")}


@app.get("/v1/ai-health")
def ai_health() -> dict[str, Any]:
    if not groq_enabled():
        return {"configured": False, "available": False, "error": "missing_key"}
    result = groq_generate("Reply with exactly OK.", "Connection test", max_tokens=5)
    return {"configured": True, "available": bool(result), "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), "error": LAST_GROQ_ERROR}


@app.post("/v1/context")
def push_context(body: ContextPush) -> dict[str, Any]:
    if body.scope not in VALID_SCOPES:
        raise HTTPException(400, detail={"accepted": False, "reason": "invalid_scope"})
    with LOCK, _db() as conn:
        row = conn.execute("SELECT version FROM contexts WHERE scope=? AND context_id=?", (body.scope, body.context_id)).fetchone()
        if row and row["version"] >= body.version:
            return {"accepted": False, "reason": "stale_version", "current_version": row["version"]}
        conn.execute("INSERT INTO contexts(scope,context_id,version,payload,delivered_at) VALUES(?,?,?,?,?) ON CONFLICT(scope,context_id) DO UPDATE SET version=excluded.version,payload=excluded.payload,delivered_at=excluded.delivered_at", (body.scope, body.context_id, body.version, json.dumps(body.payload, separators=(",", ":")), body.delivered_at))
    ack = hashlib.sha256(f"{body.scope}:{body.context_id}:{body.version}".encode()).hexdigest()[:12]
    return {"accepted": True, "ack_id": f"ack_{ack}", "stored_at": now_iso()}


@app.post("/v1/tick")
def tick(body: TickRequest) -> dict[str, list[dict[str, Any]]]:
    candidates = []
    for trigger_id in dict.fromkeys(body.available_triggers):
        trigger = get_context("trigger", trigger_id)
        if trigger and not is_expired(trigger.get("expires_at"), body.now):
            candidates.append((int(trigger.get("urgency", 0)), trigger_id, trigger))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    actions: list[dict[str, Any]] = []
    used_merchants: set[str] = set()
    with LOCK, _db() as conn:
        for _, trigger_id, trigger in candidates:
            if len(actions) >= 20:
                break
            mid = trigger.get("merchant_id")
            suppression = trigger.get("suppression_key") or f"trigger:{trigger_id}"
            if not mid or mid in used_merchants or conn.execute("SELECT 1 FROM suppressions WHERE suppression_key=?", (suppression,)).fetchone():
                continue
            merchant = get_context("merchant", mid)
            if not merchant:
                continue
            category = get_context("category", merchant.get("category_slug"))
            if not category:
                continue
            customer = get_context("customer", trigger.get("customer_id"))
            if trigger.get("scope") == "customer" and not customer:
                continue
            message, cta, rationale = compose(trigger, merchant, category, customer)
            digest = hashlib.sha256(f"{mid}:{trigger_id}:{suppression}".encode()).hexdigest()[:10]
            conv_id = f"conv_{digest}"
            action = {"conversation_id": conv_id, "merchant_id": mid, "customer_id": trigger.get("customer_id"), "send_as": "merchant_on_behalf" if customer else "vera", "trigger_id": trigger_id, "template_name": f"vera_{trigger.get('kind', 'update')}_v1", "template_params": [], "body": clamp(message), "cta": cta, "suppression_key": suppression, "rationale": rationale}
            actions.append(action)
            used_merchants.add(mid)
            ts = now_iso()
            conn.execute("INSERT OR REPLACE INTO conversations(conversation_id,merchant_id,customer_id,trigger_id,trigger_kind,last_body,auto_reply_count,status,updated_at) VALUES(?,?,?,?,?,?,0,'open',?)", (conv_id, mid, trigger.get("customer_id"), trigger_id, trigger.get("kind", "update"), action["body"], ts))
            conn.execute("INSERT OR REPLACE INTO suppressions(suppression_key,sent_at) VALUES(?,?)", (suppression, ts))
            conn.execute("INSERT INTO turns(conversation_id,role,body,created_at) VALUES(?,?,?,?)", (conv_id, "vera", action["body"], ts))
    return {"actions": actions}


AUTO_PATTERNS = ("thank you for contacting", "thanks for contacting", "we have received your message", "will get back to you", "business hours", "away right now", "auto-reply", "automated message")
STOP_PATTERNS = ("stop messaging", "unsubscribe", "not interested", "do not contact", "don't contact", "no thanks", "remove me")
GO_PATTERNS = ("go ahead", "let's do it", "lets do it", "yes do it", "yes, do", "please proceed", "i want to join", "sign me up", "confirm")
LATER_PATTERNS = ("later", "busy", "call me tomorrow", "not now", "give me time")
OFF_TOPIC = ("gst", "tax filing", "income tax", "loan", "passport")


@app.post("/v1/reply")
def reply(body: ReplyRequest) -> dict[str, Any]:
    text = re.sub(r"\s+", " ", body.message).strip()
    lower = text.lower()
    with LOCK, _db() as conn:
        conv = conn.execute("SELECT * FROM conversations WHERE conversation_id=?", (body.conversation_id,)).fetchone()
        if not conv:
            raise HTTPException(404, detail="unknown_conversation")
        if conv["status"] == "ended":
            return {"action": "end", "rationale": "Conversation was already closed; no further message is appropriate"}
        ts = now_iso()
        conn.execute("INSERT INTO turns(conversation_id,role,body,created_at) VALUES(?,?,?,?)", (body.conversation_id, body.from_role, text, body.received_at))
        if any(x in lower for x in STOP_PATTERNS) or any(x in lower for x in ("idiot", "stupid", "fuck", "abuse")):
            conn.execute("UPDATE conversations SET status='ended',updated_at=? WHERE conversation_id=?", (ts, body.conversation_id))
            return {"action": "end", "rationale": "Explicit opt-out or hostile response detected; ending respectfully and preventing escalation"}
        if any(x in lower for x in AUTO_PATTERNS):
            count = conv["auto_reply_count"] + 1
            status = "ended" if count >= 2 else "waiting"
            conn.execute("UPDATE conversations SET auto_reply_count=?,status=?,updated_at=? WHERE conversation_id=?", (count, status, ts, body.conversation_id))
            if count >= 2:
                return {"action": "end", "rationale": "Repeated canned auto-reply detected; ending to avoid an automation loop"}
            return {"action": "wait", "wait_seconds": 14400, "rationale": "Canned WhatsApp Business auto-reply detected; backing off four hours for the owner"}
        if any(x in lower for x in LATER_PATTERNS):
            conn.execute("UPDATE conversations SET status='waiting',updated_at=? WHERE conversation_id=?", (ts, body.conversation_id))
            return {"action": "wait", "wait_seconds": 1800, "rationale": "Merchant asked for time; backing off for 30 minutes"}
        trigger = get_context("trigger", conv["trigger_id"]) or {}
        merchant = get_context("merchant", conv["merchant_id"]) or {}
        customer = get_context("customer", conv["customer_id"])
        name = (customer or {}).get("identity", {}).get("name") or first_name(merchant)
        history = [dict(row) for row in conn.execute(
            "SELECT role,body FROM turns WHERE conversation_id=? ORDER BY id DESC LIMIT 6",
            (body.conversation_id,),
        ).fetchall()][::-1]
        ai_response = groq_generate(
            "You are Vera, a concise WhatsApp business assistant. Reply naturally in under 500 characters. "
            "Use ONLY the supplied context and conversation. Never invent facts, URLs, results, or completed "
            "actions. Answer the user's actual message, stay on the original business goal, and end with at "
            "most one useful question. Return only the reply.",
            json.dumps({"trigger": trigger, "merchant": merchant, "customer": customer,
                        "conversation": history, "latest_message": text}, ensure_ascii=False),
            max_tokens=180,
        )
        if ai_response:
            response = clamp(ai_response)
            rationale = "Groq generated a contextual follow-up grounded in the stored conversation and business data"
            conn.execute("UPDATE conversations SET last_body=?,status='open',updated_at=? WHERE conversation_id=?", (response, ts, body.conversation_id))
            conn.execute("INSERT INTO turns(conversation_id,role,body,created_at) VALUES(?,?,?,?)", (body.conversation_id, "vera", response, ts))
            return {"action": "send", "body": response, "cta": "binary", "rationale": rationale}
        if any(x in lower for x in GO_PATTERNS):
            kind = conv["trigger_kind"].replace("_", " ")
            response = f"Done, {name} — I’m moving ahead with the {kind} action using the details already shared. I’ll prepare the first ready-to-review draft now."
            rationale = "Explicit commitment detected; switches immediately from qualification to action execution"
        elif any(x in lower for x in OFF_TOPIC):
            response = f"I can’t advise on that—that is best handled by the relevant professional. Coming back to this {conv['trigger_kind'].replace('_', ' ')}: reply YES if you want me to prepare the promised draft."
            rationale = "Politely declines an out-of-scope request and returns to the grounded conversation goal"
        elif "?" in text:
            response = f"Good question, {name}. I’ll keep this grounded in the details we have and won’t assume anything missing. The next useful step is the draft I offered—shall I prepare it?"
            rationale = "Acknowledges the question, avoids fabrication, and offers the original low-friction next step"
        else:
            response = f"Thanks, {name}. I’ve noted that. Shall I turn the {conv['trigger_kind'].replace('_', ' ')} update into the practical draft I offered? Reply YES."
            rationale = "Acknowledges the reply and returns to one clear next action without repeating the original pitch"
        response = clamp(response)
        conn.execute("UPDATE conversations SET last_body=?,status='open',updated_at=? WHERE conversation_id=?", (response, ts, body.conversation_id))
        conn.execute("INSERT INTO turns(conversation_id,role,body,created_at) VALUES(?,?,?,?)", (body.conversation_id, "vera", response, ts))
        return {"action": "send", "body": response, "cta": "binary", "rationale": rationale}
