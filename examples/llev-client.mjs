import { pathToFileURL } from 'node:url';

// Minimal LLEV client (Node 18+ / Next.js server routes). Server-side only: never ship the key to apps.
//   import { decide, feedback } from './llev-client.mjs'
const URL = process.env.LLEV_URL || 'http://localhost:8088';
const KEY = process.env.LLEV_API_KEY;
const TIMEOUT_MS = Number(process.env.LLEV_TIMEOUT_MS || 3000);

async function call(path, body) {
  const res = await fetch(`${URL}${path}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`LLEV ${path} ${res.status}: ${JSON.stringify(data.detail ?? data)}`);
  return data;
}

/** @returns {Promise<{id: string, answers: Record<string, any>, latency_ms: number}>} */
export const decide = (state, questions, opts = {}) => call('/v1/decide', { state, questions, ...opts });

/** label: option key (choice) | level index (score) | boolean (noul) | string[] (multi) */
export const feedback = (id, key, label, source) => call('/v1/feedback', { id, key, label, source });

// Example: triage, and only automate when confident. LLEV being down must never block the ticket.
if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const { id, answers } = await decide(
      { message: 'I was charged twice for my subscription, please refund one payment.' },
      {
        team: { type: 'choice', task: 'ticket.team', instructions: 'Which support team should handle this ticket',
                criteria: { payment: 'Fares, refunds', safety: 'Danger, harassment', other: 'Anything else' } },
        danger: { type: 'noul', tier: 'accurate', instructions: "The writer describes a risk to someone's physical safety" },
      },
    );
    const { team, danger } = answers;
    const action = danger.noul > 0.5 ? 'ESCALATE to safety now'
      : team.confidence >= 0.8 ? `auto-route to ${team.choice}`
      : 'send to manual triage queue';
    console.log({ id, team: team.choice, confidence: team.confidence.toFixed(2), danger: danger.noul, action });
  } catch (err) {
    console.error('LLEV unavailable, falling back to manual triage:', err.message);
  }
}
