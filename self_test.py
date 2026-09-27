"""Dependency-light contract and behavior checks for the Vera engine."""
import os
import tempfile

db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
db.close()
os.environ["VERA_DB_PATH"] = db.name

from fastapi.testclient import TestClient
import bot
from bot import app

client = TestClient(app)


def push(scope, context_id, payload, version=1):
    return client.post("/v1/context", json={"scope": scope, "context_id": context_id, "version": version, "payload": payload, "delivered_at": "2026-04-26T10:00:00Z"})


category = {"slug": "dentists", "offer_catalog": [{"title": "Dental Cleaning @ ₹299"}], "digest": [{"id": "new", "title": "New recall evidence from a 2,100-patient trial", "source": "JIDA 2026 p.14"}]}
merchant = {"merchant_id": "m1", "category_slug": "dentists", "identity": {"name": "Dr. Meera's Dental Clinic", "locality": "Lajpat Nagar"}, "performance": {"views": 2410, "calls": 18}, "offers": [{"title": "Dental Cleaning @ ₹299", "status": "active"}]}
trigger = {"id": "t1", "scope": "merchant", "kind": "research_digest", "merchant_id": "m1", "customer_id": None, "payload": {"top_item_id": "new"}, "urgency": 2, "suppression_key": "research:test", "expires_at": "2027-01-01T00:00:00Z"}

assert push("category", "dentists", category).status_code == 200
assert push("merchant", "m1", merchant).status_code == 200
assert push("trigger", "t1", trigger).status_code == 200
assert push("trigger", "t1", trigger).json()["reason"] == "stale_version"

result = client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z", "available_triggers": ["t1"]}).json()
assert len(result["actions"]) == 1
action = result["actions"][0]
assert "2,100-patient" in action["body"] and "JIDA" in action["body"]
assert client.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z", "available_triggers": ["t1"]}).json() == {"actions": []}

auto = client.post("/v1/reply", json={"conversation_id": action["conversation_id"], "merchant_id": "m1", "message": "Thank you for contacting us. We will get back to you.", "received_at": "2026-04-26T10:31:00Z", "turn_number": 2}).json()
assert auto["action"] == "wait"
auto2 = client.post("/v1/reply", json={"conversation_id": action["conversation_id"], "merchant_id": "m1", "message": "Thank you for contacting us. We will get back to you.", "received_at": "2026-04-26T10:32:00Z", "turn_number": 3}).json()
assert auto2["action"] == "end"
assert client.get("/v1/healthz").json()["contexts_loaded"] == {"category": 1, "merchant": 1, "customer": 0, "trigger": 1}

# Verify the AI path independently without making a paid network call.
original_groq = bot.groq_generate
bot.groq_generate = lambda system, user, max_tokens=240: "AI-generated grounded draft for Dr. Meera. Shall I prepare it?"
message, _, rationale = bot.compose(trigger, merchant, category, None)
assert message.startswith("AI-generated") and rationale.startswith("Groq generated")
assert bot.is_numerically_grounded("Offer at ₹299", category)
assert not bot.is_numerically_grounded("Offer at ₹349", category)
bot.groq_generate = original_groq
print("All Vera contract tests passed")
