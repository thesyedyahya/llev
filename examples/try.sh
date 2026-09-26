#!/usr/bin/env bash
# End-to-end API walkthrough: every question type, feedback, memory.
#   LLEV_URL=http://localhost:8088 LLEV_KEY=... ./examples/try.sh
set -euo pipefail
URL="${LLEV_URL:-http://localhost:8088}"
KEY="${LLEV_KEY:?set LLEV_KEY}"
J=(-sS -H "Authorization: Bearer $KEY" -H "Content-Type: application/json")
pp() { python3 -m json.tool; }

echo "== health"; curl -sS "$URL/health"; echo

echo "== 1. choice + score + noul + multi in one request"
RESP=$(curl "${J[@]}" "$URL/v1/decide" -d '{
  "state": {"message": "The courier left my parcel outside, it got stolen, and he laughed when I called. I want my money back.", "plan": "pro"},
  "questions": {
    "team": {"type": "choice", "task": "support.route",
             "instructions": "Which team should handle this customer support message",
             "criteria": {"billing": "Charges, refunds, invoices", "technical": "Bugs, errors, login problems",
                          "shipping": "Delivery, lost or damaged packages", "trust_safety": "Harassment, threats, fraud",
                          "other": "None of the above"}},
    "urgency": {"type": "score", "instructions": "How urgently a human needs to respond",
                "criteria": ["Days", "Today", "Within the hour", "Immediately: safety risk or critical outage"]},
    "danger": {"type": "noul", "tier": "accurate", "instructions": "The writer describes a risk to someone'"'"'s physical safety"},
    "tags": {"type": "multi", "instructions": "What does the message mention",
             "criteria": {"refund": "wants money back", "theft": "something was stolen", "rude_staff": "rude employee or courier"}}
  }
}')
echo "$RESP" | pp
ID=$(echo "$RESP" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')

echo "== 2. feedback: tell LLEV the correct answer (label = option key / level index / true|false / list)"
curl "${J[@]}" "$URL/v1/feedback" -d "{\"id\": \"$ID\", \"key\": \"team\", \"label\": \"shipping\", \"source\": \"try.sh\"}"; echo

echo "== 3. memory per task"
curl "${J[@]}" "$URL/v1/memory"; echo

echo "== 4. Jev-compatible path"
curl "${J[@]}" "$URL/v1/systemone" -d '{"state": "Great service, very polite staff!",
  "questions": {"abusive": {"type": "noul", "instructions": "The review contains insults or abusive language"}}}' | pp

echo "== 5. errors: no key -> 401, bad label -> 422"
curl -s -o /dev/null -w "%{http_code}\n" "$URL/v1/decide" -H "Content-Type: application/json" -d '{"state":"x","questions":{"n":{"type":"noul","instructions":"x"}}}'
curl -s -o /dev/null -w "%{http_code}\n" "${J[@]}" "$URL/v1/feedback" -d "{\"id\": \"$ID\", \"key\": \"team\", \"label\": \"not_an_option\"}"
