import inspect
import json
import math
import os
import platform
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Set, Tuple

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from llama_cpp import Llama, llama_chat_format

DEFAULT_SYSTEM_PROMPT = "You are an assistant."
SMALL_MODEL_PATH = os.environ.get("SMALL_MODEL_PATH", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
LARGE_MODEL_PATH = os.environ.get("LARGE_MODEL_PATH", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
DEFAULT_PPL_THRESHOLD = float(os.environ.get("PPL_THRESHOLD", "5.0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
_CPU_COUNT = os.cpu_count() or 1
DEFAULT_THREADS = int(os.environ.get("LLAMA_THREADS", str(max(_CPU_COUNT - 1, 1))))
DEFAULT_BATCH_SIZE = int(os.environ.get("LLAMA_BATCH", "512"))
_CACHE_PROMPT_ENV = os.environ.get("LLAMA_CACHE_PROMPT")
if _CACHE_PROMPT_ENV is None:
    DEFAULT_CACHE_PROMPT = platform.system().lower() != "darwin"
else:
    DEFAULT_CACHE_PROMPT = _CACHE_PROMPT_ENV != "0"
DEFAULT_N_GPU_LAYERS = int(os.environ.get(
    "N_GPU_LAYERS",
    "-1" if platform.system().lower() == "darwin" else "0",
))
STOP_TOKENS = {"<|im_end|>", "</s>", "<|endoftext|>", "<|eot_id|>"}
SMALL_CHAT_FORMAT = os.environ.get("SMALL_CHAT_FORMAT", "llama-3")
LARGE_CHAT_FORMAT = os.environ.get("LARGE_CHAT_FORMAT", "qwen")

CHAT_FORMATTERS = {
    "llama-3": llama_chat_format.format_llama3,
    "qwen": llama_chat_format.format_qwen,
    "chatml": llama_chat_format.format_chatml,
}


def _resolve_chat_formatter(name: str):
    try:
        return CHAT_FORMATTERS[name]
    except KeyError as exc:  # pragma: no cover - configuration error
        raise ValueError(f"Unsupported chat format '{name}'") from exc


SMALL_CHAT_FORMATTER = _resolve_chat_formatter(SMALL_CHAT_FORMAT)
LARGE_CHAT_FORMATTER = _resolve_chat_formatter(LARGE_CHAT_FORMAT)
LLAMA_SUPPORTS_CACHE_PROMPT = "cache_prompt" in inspect.signature(Llama.__call__).parameters


def _reset_llm_cache(llm: Llama) -> None:
    reset = getattr(llm, "reset", None)
    if callable(reset):
        reset()


@dataclass
class GenerationContext:
    small_prompt_base: str
    large_prompt_base: str
    small_stop_tokens: Set[str]
    large_stop_tokens: Set[str]
    combined_stop_tokens: Set[str]
    perplexity_threshold: float

app = Flask(__name__)
conversations: Dict[str, List[Dict[str, str]]] = {}
generation_lock = threading.Lock()


def load_model(path: str, *, logits_all: bool = False) -> Llama:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    threads_env = os.environ.get("LLAMA_THREADS")
    batch_env = os.environ.get("LLAMA_BATCH")
    n_gpu_layers_env = os.environ.get("N_GPU_LAYERS")

    threads = int(threads_env) if threads_env else DEFAULT_THREADS
    n_batch = int(batch_env) if batch_env else DEFAULT_BATCH_SIZE
    n_gpu_layers = int(n_gpu_layers_env) if n_gpu_layers_env else DEFAULT_N_GPU_LAYERS

    kwargs = {
        "model_path": path,
        "chat_format": None,
        "logits_all": logits_all,
        "n_threads": threads,
        "n_batch": n_batch,
        "verbose": False,
    }

    if n_gpu_layers != 0:
        kwargs["n_gpu_layers"] = n_gpu_layers

    try:
        return Llama(**kwargs)
    except TypeError:
        kwargs.pop("n_gpu_layers", None)
        return Llama(**kwargs)


try:
    small_model = load_model(SMALL_MODEL_PATH, logits_all=True)
    large_model = load_model(LARGE_MODEL_PATH)
except Exception as exc:  # pragma: no cover - startup failure is fatal
    raise RuntimeError(f"Failed to load models: {exc}") from exc


def get_next_token_and_logprob(
    llm: Llama,
    prompt: str,
    extra_stop_tokens: Set[str],
    *,
    need_logprob: bool,
) -> Tuple[str, Optional[float], Optional[str]]:
    if LLAMA_SUPPORTS_CACHE_PROMPT:
        if DEFAULT_CACHE_PROMPT:
            _reset_llm_cache(llm)
    kwargs = {
        "max_tokens": 1,
        "echo": False,
        "stop": list(STOP_TOKENS | extra_stop_tokens),
    }
    if need_logprob:
        kwargs["logprobs"] = 1
    if LLAMA_SUPPORTS_CACHE_PROMPT and DEFAULT_CACHE_PROMPT:
        kwargs["cache_prompt"] = True
    out = llm(prompt, **kwargs)
    choice = out["choices"][0]
    logprob = None
    if need_logprob:
        logprobs = choice.get("logprobs")
        if logprobs:
            token_logprobs = logprobs.get("token_logprobs")
            if token_logprobs:
                logprob = token_logprobs[0]
    return choice.get("text", ""), logprob, choice.get("finish_reason")
    kwargs = {
        "max_tokens": 1,
        "logprobs": 1,
        "echo": False,
        "stop": list(STOP_TOKENS | extra_stop_tokens),
    }
    if LLAMA_SUPPORTS_CACHE_PROMPT and DEFAULT_CACHE_PROMPT:
        kwargs["cache_prompt"] = True
    out = llm(prompt, **kwargs)
    choice = out["choices"][0]
    logprob = None
    logprobs = choice.get("logprobs")
    if logprobs:
        token_logprobs = logprobs.get("token_logprobs")
        if token_logprobs:
            logprob = token_logprobs[0]
    return choice.get("text", ""), logprob, choice.get("finish_reason")


def perplexity_from_logprob(logprob: float) -> float:
    return math.exp(-logprob)


def build_model_prompt(
    formatter,
    system_prompt: str,
    messages: List[Dict[str, str]],
) -> Tuple[str, Set[str]]:
    chat_messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}, *messages]
    formatted = formatter(chat_messages)
    extra_stops: Set[str] = set()
    if formatted.stop is not None:
        if isinstance(formatted.stop, str):
            extra_stops.add(formatted.stop)
        else:
            extra_stops.update(formatted.stop)
    return formatted.prompt, extra_stops


def _build_generation_context(
    history: List[Dict[str, str]],
    system_prompt: str,
    perplexity_threshold: Optional[float] = None,
) -> GenerationContext:
    threshold = DEFAULT_PPL_THRESHOLD if perplexity_threshold is None else float(perplexity_threshold)
    small_prompt_base, small_stop_tokens = build_model_prompt(
        SMALL_CHAT_FORMATTER,
        system_prompt,
        history,
    )
    large_prompt_base, large_stop_tokens = build_model_prompt(
        LARGE_CHAT_FORMATTER,
        system_prompt,
        history,
    )
    combined_stop_tokens = set(STOP_TOKENS)
    combined_stop_tokens.update(small_stop_tokens)
    combined_stop_tokens.update(large_stop_tokens)
    return GenerationContext(
        small_prompt_base=small_prompt_base,
        large_prompt_base=large_prompt_base,
        small_stop_tokens=small_stop_tokens,
        large_stop_tokens=large_stop_tokens,
        combined_stop_tokens=combined_stop_tokens,
        perplexity_threshold=threshold,
    )


def generate_token_with_fallback(
    small_prompt: str,
    large_prompt: str,
    small_stop_tokens: Set[str],
    large_stop_tokens: Set[str],
    threshold: float,
) -> Tuple[str, Optional[str], bool]:
    token, logprob, finish_reason = get_next_token_and_logprob(
        small_model,
        small_prompt,
        small_stop_tokens,
        need_logprob=True,
    )
    if logprob is None:
        perplexity = float("inf")
    else:
        perplexity = perplexity_from_logprob(logprob)

    if perplexity > threshold:
        token, _, finish_reason = get_next_token_and_logprob(
            large_model,
            large_prompt,
            large_stop_tokens,
            need_logprob=False,
        )
        return token, finish_reason, True
    return token, finish_reason, False


def clean_response_text(text: str, extra_stop_tokens: Optional[Set[str]] = None) -> str:
    cleaned = text
    all_stops = STOP_TOKENS if extra_stop_tokens is None else STOP_TOKENS | extra_stop_tokens
    for stop in all_stops:
        cleaned = cleaned.replace(stop, "")
    return cleaned.strip()


def _partial_stop_length(text: str, stop: str) -> int:
    max_len = min(len(text), len(stop) - 1)
    partial = 0
    for length in range(1, max_len + 1):
        if text.endswith(stop[:length]):
            partial = length
    return partial


def _longest_partial_stop_suffix(text: str, stop_tokens: Set[str]) -> int:
    max_partial = 0
    for stop in stop_tokens:
        partial = _partial_stop_length(text, stop)
        if partial > max_partial:
            max_partial = partial
    return max_partial


def stream_response_tokens(
    history: List[Dict[str, str]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    context: Optional[GenerationContext] = None,
    perplexity_threshold: Optional[float] = None,
) -> Iterator[Tuple[str, bool]]:
    ctx = context or _build_generation_context(history, system_prompt, perplexity_threshold)
    raw_response = ""
    visible_response = ""
    pending_from_large = False
    for _ in range(MAX_TOKENS):
        prompt_small = ctx.small_prompt_base + raw_response
        prompt_large = ctx.large_prompt_base + raw_response
        token, finish_reason, from_large = generate_token_with_fallback(
            prompt_small,
            prompt_large,
            ctx.small_stop_tokens,
            ctx.large_stop_tokens,
            ctx.perplexity_threshold,
        )
        if not token and finish_reason:
            break
        if not token:
            continue
        pending_from_large = pending_from_large or from_large

        raw_response += token

        done = False
        stop_index: Optional[int] = None
        for stop in ctx.combined_stop_tokens:
            idx = raw_response.find(stop)
            if idx != -1 and (stop_index is None or idx < stop_index):
                stop_index = idx
        if stop_index is not None:
            raw_response = raw_response[:stop_index]
            done = True

        visible_candidate = raw_response
        partial_len = 0 if done else _longest_partial_stop_suffix(raw_response, ctx.combined_stop_tokens)
        if partial_len:
            visible_candidate = visible_candidate[:-partial_len]

        if len(visible_candidate) > len(visible_response):
            delta = visible_candidate[len(visible_response) :]
            if delta:
                yield delta, pending_from_large
                pending_from_large = False
            visible_response = visible_candidate
        elif len(visible_candidate) < len(visible_response):
            visible_response = visible_candidate

        if done or (finish_reason and finish_reason != "length"):
            break


def generate_response(
    history: List[Dict[str, str]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    perplexity_threshold: Optional[float] = None,
) -> str:
    response_parts = []
    ctx = _build_generation_context(history, system_prompt, perplexity_threshold)
    for delta, _ in stream_response_tokens(
        history,
        system_prompt,
        context=ctx,
        perplexity_threshold=perplexity_threshold,
    ):
        response_parts.append(delta)
    response = "".join(response_parts)
    return clean_response_text(response, ctx.combined_stop_tokens)


@app.route("/", methods=["GET"])
def serve_index() -> str:
    return send_from_directory(os.getcwd(), "index.html")


@app.route("/api/chat", methods=["POST"])
def chat() -> Response:
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message is required."}), 400

    conversation_id = data.get("conversation_id")
    if not conversation_id or conversation_id not in conversations:
        conversation_id = conversation_id or str(uuid.uuid4())
        conversations.setdefault(conversation_id, [])

    def stream() -> Iterator[str]:
        history = conversations[conversation_id]
        buffered_response = ""
        try:
            with generation_lock:
                history.append({"role": "user", "content": message})
                threshold = data.get("perplexity_threshold")
                if threshold is not None:
                    try:
                        threshold = float(threshold)
                    except (TypeError, ValueError):  # pragma: no cover - input validation fallback
                        threshold = None
                ctx = _build_generation_context(history, DEFAULT_SYSTEM_PROMPT, threshold)
                yield json.dumps({"conversation_id": conversation_id, "event": "start"}) + "\n"
                for delta, from_large in stream_response_tokens(
                    history,
                    DEFAULT_SYSTEM_PROMPT,
                    context=ctx,
                    perplexity_threshold=threshold,
                ):
                    buffered_response += delta
                    yield json.dumps({
                        "conversation_id": conversation_id,
                        "delta": delta,
                        "from_large": from_large,
                    }) + "\n"

                cleaned = clean_response_text(buffered_response, ctx.combined_stop_tokens)
                history.append({"role": "assistant", "content": cleaned})
        except Exception as exc:  # pragma: no cover - runtime failure path
            app.logger.exception("Generation failed")
            failed_history = conversations.get(conversation_id, [])
            if failed_history and failed_history[-1].get("role") == "user":
                failed_history.pop()
            if not failed_history:
                conversations.pop(conversation_id, None)
            yield json.dumps({
                "conversation_id": conversation_id,
                "error": f"Generation failed: {exc}",
            }) + "\n"
            return

        yield json.dumps({
            "conversation_id": conversation_id,
            "done": True,
            "response": history[-1]["content"],
        }) + "\n"

    return Response(stream_with_context(stream()), mimetype="application/x-ndjson")


@app.route("/api/conversations/<conversation_id>", methods=["DELETE"])
def delete_conversation(conversation_id: str) -> Response:
    with generation_lock:
        removed = conversations.pop(conversation_id, None)
    return jsonify({"cleared": bool(removed)})


@app.route("/api/health", methods=["GET"])
def health() -> "flask.Response":
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, threaded=True)
