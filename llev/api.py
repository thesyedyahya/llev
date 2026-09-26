import hmac
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from . import __version__
from .config import Settings, get_settings
from .engine import ContextOverflow, Engine, FakeEngine, LlamaEngine
from .memory import ExampleMemory
from .models import REGISTRY, local_path
from .schemas import DecideRequest, DecideResponse, FeedbackRequest, FeedbackResponse
from .service import LLEV
from .store import DecisionStore
from .templates import get_template


def _load_engine(s: Settings, key: str | None, path: str | None, template: str | None) -> Engine | None:
    if not key and not path:
        return None
    spec = REGISTRY.get(key or "")
    if s.engine == "fake":
        return FakeEngine(name=key or "fake")
    model_path = path or (local_path(s.models_dir, key) if key else None)
    if not model_path:
        raise RuntimeError(f"model {key!r} not found in {s.models_dir}/; run scripts/download_model.py {key}")
    tpl = template or (spec.template if spec else None)
    if not tpl:
        raise RuntimeError(f"set a template for custom model {model_path}")
    return LlamaEngine(key or Path(model_path).stem, str(model_path), get_template(tpl),
                       s.n_ctx, s.n_threads, s.n_gpu_layers)


def build_service(s: Settings, primary: Engine | None = None, escalation: Engine | None = None) -> LLEV:
    data = Path(s.data_dir)
    primary = primary or _load_engine(s, s.model, s.model_path, s.template)
    if escalation is None:
        escalation = _load_engine(s, s.escalate_model, s.escalate_model_path, s.escalate_template)
    return LLEV(s, primary, escalation,
                ExampleMemory(data / "memory.jsonl", s.fewshot_max_per_task),
                DecisionStore(data))


def create_app(settings: Settings | None = None, service: LLEV | None = None) -> FastAPI:
    s = settings or get_settings()
    if not s.api_key_list and not s.allow_no_auth:
        raise RuntimeError("set LLEV_API_KEYS (or LLEV_ALLOW_NO_AUTH=true for local dev)")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not hasattr(app.state, "llev"):
            app.state.llev = build_service(s)  # loads the model(s): slow, so only at startup
        yield

    app = FastAPI(title="LLEV", version=__version__, lifespan=lifespan)
    if service is not None:
        app.state.llev = service

    def auth(request: Request) -> None:
        if not s.api_key_list:
            return
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        if not any(hmac.compare_digest(token, k) for k in s.api_key_list):
            raise HTTPException(401, "invalid api key")

    def svc(request: Request) -> LLEV:
        return request.app.state.llev

    @app.get("/health")
    def health(request: Request):
        llev = svc(request)
        return {"status": "ok", "version": __version__, "model": llev.primary.name,
                "escalation": llev.escalation.name if llev.escalation else None}

    @app.post("/v1/decide", response_model=DecideResponse, response_model_exclude_none=True,
              dependencies=[Depends(auth)])
    async def decide(req: DecideRequest, request: Request):
        if len(req.questions) > s.max_questions:
            raise HTTPException(422, f"at most {s.max_questions} questions per request")
        state_len = len(req.state) if isinstance(req.state, str) else len(str(req.state))
        if state_len > s.max_state_chars:
            raise HTTPException(413, f"state exceeds {s.max_state_chars} chars")
        try:
            return await run_in_threadpool(svc(request).decide, req)
        except ContextOverflow as e:
            raise HTTPException(413, str(e)) from e

    # Jev-compatible path, so existing Jev-style clients can point here unchanged.
    app.add_api_route("/v1/systemone", decide, methods=["POST"], response_model=DecideResponse,
                      response_model_exclude_none=True, dependencies=[Depends(auth)])

    @app.post("/v1/feedback", response_model=FeedbackResponse, dependencies=[Depends(auth)])
    async def feedback(fb: FeedbackRequest, request: Request):
        try:
            learned, detail = await run_in_threadpool(svc(request).feedback, fb.id, fb.key, fb.label, fb.source)
        except KeyError as e:
            raise HTTPException(404, e.args[0]) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return FeedbackResponse(ok=True, learned=learned, detail=detail)

    playground = (Path(__file__).parent / "playground.html").read_text(encoding="utf-8")

    @app.get("/playground", response_class=HTMLResponse, include_in_schema=False)
    def playground_page():
        # Static page only; every API call it makes still needs the key.
        return playground

    @app.get("/v1/memory", dependencies=[Depends(auth)])
    def memory(request: Request):
        return {"tasks": svc(request).memory.stats()}

    return app
