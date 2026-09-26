# Security

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/thesyedyahya/llev/security/advisories/new)
rather than in public issues.

Deployment notes:

- LLEV refuses to start without `LLEV_API_KEYS` unless `LLEV_ALLOW_NO_AUTH=true` is set, and that flag is for local dev only.
- Call LLEV from your backend only. Never ship an API key to browsers or mobile apps.
- With `LLEV_LOG_STATE=true` (the default), request text is stored in `LLEV_DATA_DIR` so feedback can build
  memory and training data. Treat that directory as sensitive, or turn logging off.
- Decisions about safety or money should use `"tier": "accurate"` and a human in the loop. Small models can be
  confidently wrong.
