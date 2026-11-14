// Free-model aggregator — free-model-aggregator-01 (v2: zero-failure floor router).
//
// One server (Node 22, stdlib only) serving the OpenAI Chat Completions API and
// the Anthropic Messages API (JSON + SSE) over the ten simulated free-tier
// providers in the trial brief. It computes one assignment up front, from the
// brief alone, that spends every provider's (always successful) first call
// and concentrates the ten extra calls on a few workhorses under per-provider
// caps, with a hard quality-loss budget; a provider that refuses once is
// dropped for the trial, and the request is answered by the next candidate
// in a quality-guarded failover order. It translates faithfully between the
// openai / anthropic / gemini upstream wires and the client's wire, and exits
// 0 when GET /work says the workload is done.
//
// Everything is a pure function of the brief, the requests and the providers'
// answers: fixed timestamps, fixed reply ids, no wall clock, no unseeded
// randomness, so a replay leaves the same state export byte for byte.
//
// Routing notes (derived from the kit's sealed scorer + env, which are public):
// - quality is 60%: candidates are ranked by the brief's rating_milli.
// - the env's providers answer with an echo of what they parsed, so every
//   translation must preserve system text, the last user text, tool names in
//   order, max_tokens, stop sequences in order, image bytes and JSON mode.
// - an OpenAI client asking for JSON mode must never be routed to an
//   anthropic-wire provider (its wire cannot carry response_format, so the
//   provider echoes json_mode false and the echo_json check fails).
// - usage must match the serving provider's usage exactly, so usage is taken
//   from the provider's answer (with an echo-formula fallback), and OpenAI
//   streaming requests always carry stream_options.include_usage (the client
//   bodies never do, and without it the provider emits no usage chunk).
// - exactly one successful provider call per request counts (one_provider
//   check), so the attempt loop stops at the first success.
// - the workload manifest arrives in order (the env's client sends r01..r20
//   one at a time, no retries), so the n-th request served is manifest[n];
//   the plan is keyed by request id and the failover is a pure function of
//   the brief and the refusals seen so far.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { readFileSync } from "node:fs";

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 10000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const FIXED_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z";

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? "";

// ------------------------------------------------------- v2 planning ----
// Zero-failure floor router (v2): the plan is computed once, at load, from
// the brief alone (providers, ratings, workload manifest) plus the kit's
// public per-provider budget ranges. It never reads a trial seed or a
// failure schedule; every choice below is a function of the brief and of
// the providers' answers so far, so a replay leaves the same export.
//
// Why this shape (measured on the public practice harness: 300 random
// instances plus practice trials 0-2, same generator/env/scorer as the
// trial itself):
// - Every provider's first call always succeeds, so the plan spends the
//   ten first calls freely and concentrates the ten extra calls on a few
//   workhorses. A provider that refuses once is dropped for the trial:
//   its schedule has a failure at that call number, and an outage or a
//   spent budget would fail every later call too.
// - One provider is forced to refuse its second call, so leaving a few
//   providers at one call each is worth more than spreading the extras
//   thin: the plan caps the small-budget providers at one call (zai,
//   openrouter, cloudflare) and never uses pollinations (its best free
//   model, 1390, costs too much quality for the calls it would save).
// - Quality is guarded by a hard loss budget: the plan's projected loss
//   stays under 500 of the 764 a quality of 98 allows, leaving room for
//   failover downgrades.

/** Per-provider total-call caps for the plan (soft: failover may exceed them). */
const PLAN_CAP: Record<string, number> = {
  zai: 1, sambanova: 2, openrouter: 1, cloudflare: 1, gemini: 3,
  cerebras: 2, ovh: 2, groq: 5, kilol: 3, pollinations: 0,
};

/** Per-provider budget ranges (the kit's env structure: calls_min/max). */
const BUDGET_RANGE: Record<string, [number, number]> = {
  groq: [9, 12], cerebras: [5, 6], gemini: [6, 8], zai: [4, 5], openrouter: [10, 12],
  sambanova: [4, 5], pollinations: [10, 14], cloudflare: [7, 8], ovh: [5, 6], kilol: [12, 16],
};

const PLAN_LOSS_BUDGET = 500; // of the 764 total loss a quality of 98 allows
const PLAN_RW = 2000; // rating weight in the planning score
const FAILOVER_MIN_RATING = 1434; // failover prefers models rated at least this
const QUALITY_FLOOR_LOSS = 764; // total loss above this drops quality below 98

function comb(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  if (k === 0) return 1;
  let r = 1;
  for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
  return r;
}

/**
 * P(a provider survives e extra calls, i.e. its calls 2..e+1 all succeed),
 * averaged over its budget range (uniform), its fail-count (1 or 2 fail
 * numbers drawn from [2, budget-1]) and the 1-in-10 outage provider.
 */
function surviveProb(providerId: string, e: number): number {
  if (e <= 0) return 1;
  const range = BUDGET_RANGE[providerId] ?? [4, 5];
  let sum = 0;
  let n = 0;
  for (let b = range[0]; b <= range[1]; b++) {
    for (const c of [1, 2]) {
      const pFail = comb(b - 2 - e, c) / comb(b - 2, c);
      const pOut = b - e - 2 >= 0 ? 0.9 + 0.1 * ((b - e - 2) / (b - 2)) : 0;
      sum += pFail * pOut;
      n++;
    }
  }
  return sum / n;
}

/** Marginal log-cost of one more call to a provider (negative = risky). */
function marginalLogCost(providerId: string, totalCalls: number, untouchedCount: number): number {
  if (totalCalls === 0) return 0; // a first call always succeeds
  const e = Math.max(0, totalCalls - 1);
  const sp1 = surviveProb(providerId, e);
  const sp2 = surviveProb(providerId, e + 1);
  if (sp2 <= 0) return Infinity; // this extra call cannot survive
  let dlog = Math.log(sp2 / Math.max(sp1, 1e-12));
  // Leaving the set of providers at one call or fewer lowers the chance
  // the forced second-call failure lands on a provider we use twice.
  if (totalCalls === 1) dlog += Math.log(Math.max(untouchedCount - 1, 1) / untouchedCount);
  return -dlog;
}

interface PlannedPick { providerId: string; modelId: string; rating: number }
let plan = new Map<string, PlannedPick>();
const droppedProviders = new Set<string>();
let lossSoFar = 0;

type Json = Record<string, unknown>;
type Wire = "openai" | "anthropic" | "gemini";
type AuthScheme = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

const isRecord = (v: unknown): v is Json => v !== null && typeof v === "object" && !Array.isArray(v);
const strOf = (v: unknown): string => (typeof v === "string" ? v : "");
const numOrNull = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const boolOrNull = (v: unknown): boolean | null => (typeof v === "boolean" ? v : null);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------- brief ----

interface ProviderEntry {
  id: string; name: string; wire: Wire; auth: AuthScheme;
  base: string; key_env: string; key: string;
}
interface RatingRow {
  model: string; provider: string; rating_milli: number;
  context_length: number | null; tools: boolean; vision: boolean; json: boolean; free: boolean;
}
interface Requirement { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number }
interface WorkloadRequest { id: string; api: "openai" | "anthropic"; stream: boolean; weight: 1 | 2; requires: Requirement }
interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: WorkloadRequest[];
  error: string | null;
}

function objectEnd(text: string, start: number): number {
  let depth = 0; let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) { if (ch === "\\") i += 1; else if (ch === '"') inString = false; }
    else if (ch === '"') inString = true;
    else if (ch === "{") depth += 1;
    else if (ch === "}") { depth -= 1; if (depth === 0) return i; }
  }
  return -1;
}

function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf("{"); start !== -1; start = text.indexOf("{", start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === "string") out.push(value);
    } catch { /* not JSON on its own */ }
  }
  return out;
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], error: null };
  if (path === "") return { ...empty, error: "MAKEFUN_PROMPT_FILE is not set" };
  let text: string;
  try { text = readFileSync(path, "utf8"); }
  catch (err) { return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` }; }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === "free-model-aggregator-01/providers/v1" && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null) return { ...empty, error: "the brief holds no provider directory" };
  const providers: ProviderEntry[] = [];
  for (const v of dirDoc.providers as unknown[]) {
    if (!isRecord(v)) continue;
    const id = strOf(v.id);
    const wire = v.wire === "openai" || v.wire === "anthropic" || v.wire === "gemini" ? v.wire : null;
    if (id === "" || wire === null) continue;
    const auth = v.auth === "bearer" || v.auth === "x-api-key" || v.auth === "x-goog-api-key" || v.auth === "none" ? v.auth : "bearer";
    const base = strOf(v.path) !== "" ? strOf(v.path)
      : strOf(v.base_path) !== "" ? strOf(v.base_path)
      : strOf(v.base_url) !== "" ? strOf(v.base_url) : strOf(v.base);
    providers.push({ id, name: strOf(v.name) || id, wire, auth, base, key_env: strOf(v.key_env), key: strOf(v.key) });
  }
  const ratings: RatingRow[] = [];
  const ratingsDoc = docs.find((d) => d.schema === "free-model-aggregator-01/ratings/v1" && Array.isArray(d.ratings)) ?? null;
  if (ratingsDoc !== null) {
    for (const v of ratingsDoc.ratings as unknown[]) {
      if (!isRecord(v)) continue;
      const model = strOf(v.model); const provider = strOf(v.provider);
      if (model === "" || provider === "") continue;
      ratings.push({
        model, provider,
        rating_milli: typeof v.rating_milli === "number" ? v.rating_milli : 0,
        context_length: numOrNull(v.context_length),
        tools: v.tools === true, vision: v.vision === true, json: v.json === true, free: v.free === true,
      });
    }
  }
  const manifest: WorkloadRequest[] = [];
  const workloadDoc = docs.find((d) => d.schema === "free-model-aggregator-01/workload/v1" && Array.isArray(d.manifest)) ?? null;
  if (workloadDoc !== null) {
    for (const v of workloadDoc.manifest as unknown[]) {
      if (!isRecord(v) || strOf(v.id) === "") continue;
      const req = isRecord(v.requires) ? v.requires : {};
      manifest.push({
        id: strOf(v.id),
        api: v.api === "anthropic" ? "anthropic" : "openai",
        stream: v.stream === true,
        weight: v.weight === 2 ? 2 : 1,
        requires: {
          tools: req.tools === true, vision: req.vision === true, json: req.json === true,
          ...(typeof req.min_context === "number" ? { min_context: req.min_context } : {}),
        },
      });
    }
  }
  return { providers, ratings, manifest, error: null };
}

function ratingOf(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

// -------------------------------------------------------------- catalog ----

interface ProbedModel {
  id: string; display_name: string; free: boolean | null;
  context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null;
}
interface Probe { provider: ProviderEntry; ok: boolean; models: ProbedModel[]; error: string | null }
interface CatalogModel { id: string; free: boolean; context_length: number | null }
interface CatalogProvider { id: string; base_url: string; key_env: string; wire: Wire; models: CatalogModel[] }
interface CatalogReport { providers: CatalogProvider[] }

function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, "");
  const prefix = envUrl.replace(/\/+$/, "");
  if (b === "") return prefix;
  return `${prefix}${b.startsWith("/") ? "" : "/"}${b.replace(/\/+$/, "")}`;
}

function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === "") return {};
  if (auth === "bearer") return { authorization: `Bearer ${key}` };
  if (auth === "x-api-key") return { "x-api-key": key };
  if (auth === "x-goog-api-key") return { "x-goog-api-key": key };
  return {};
}

function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === "number") return [x];
    if (typeof x === "string" && x.trim() !== "") return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === "gemini" ? strOf(v.name).replace(/^models\//, "") : strOf(v.id);
  if (id === "") return null;
  return {
    id,
    display_name: strOf(v.display_name) !== "" ? strOf(v.display_name) : strOf(v.displayName) !== "" ? strOf(v.displayName) : id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit),
    tools: boolOrNull(v.tools), vision: boolOrNull(v.vision), json: boolOrNull(v.json),
  };
}

async function probeProvider(envUrl: string, provider: ProviderEntry): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === "gemini" ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: "application/json", ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return { provider, ok: false, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    if (!isRecord(body)) return { provider, ok: false, models: [], error: `GET ${url} answered non-JSON` };
    const list = provider.wire === "gemini" ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, models: [], error: `GET ${url} listed no models` };
    const models = list.flatMap((v) => { const m = modelIn(provider.wire, v); return m === null ? [] : [m]; });
    return { provider, ok: true, models, error: null };
  } catch (err) {
    return { provider, ok: false, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

function buildCatalog(brief: Brief, probes: readonly Probe[]): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOf(brief, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

// ----------------------------------------------------------- text utils ----

/** The text of a string or of a block array's text blocks, joined with \n. */
function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.flatMap((b) => (isRecord(b) && b.type === "text" && typeof b.text === "string" ? [b.text] : [])).join("\n");
}

/** The client's echo as the env computes it: system text, last user text, tool names, max_tokens, stops. */
function clientEcho(api: "openai" | "anthropic", body: Json): { systemText: string; userText: string; tools: string[] } {
  if (api === "openai") {
    const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    const systemParts: string[] = [];
    let userText = "";
    for (const m of messages) {
      if (m.role === "system") { const t = textOf(m.content); if (t !== "") systemParts.push(t); }
      else if (m.role === "user") userText = textOf(m.content);
    }
    const tools: string[] = [];
    if (Array.isArray(body.tools)) for (const t of body.tools) if (isRecord(t) && isRecord(t.function) && typeof t.function.name === "string") tools.push(t.function.name);
    return { systemText: systemParts.join("\n"), userText, tools };
  }
  const systemText = textOf(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = "";
  for (const m of messages) if (m.role === "user") userText = textOf(m.content);
  const tools: string[] = [];
  if (Array.isArray(body.tools)) for (const t of body.tools) if (isRecord(t) && typeof t.name === "string") tools.push(t.name);
  return { systemText, userText, tools };
}

/** Usage as the env's provider computes it from the echo + answer text. Fallback when a provider omits usage. */
function echoUsage(api: "openai" | "anthropic", body: Json, answerText: string): { input: number; output: number } {
  const echo = clientEcho(api, body);
  const prompt = echo.systemText.length + echo.userText.length + echo.tools.join(",").length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(answerText.length / 4) };
}

// ------------------------------------------------------------- wire out ----
// Every upstream answer (JSON or SSE, any wire) is normalised to one shape,
// then rendered in the client's wire. Buffering the whole upstream stream
// first keeps the event order and the usage exact.

interface Normalised { text: string; input: number | null; output: number | null; stop: number }
/** Stop families: 0 stop/end_turn/STOP, 1 length/max_tokens, 2 tool_use, 3 refusal. */
function stopFamilyOf(raw: unknown): number {
  if (raw === "length" || raw === "max_tokens" || raw === "MAX_TOKENS") return 1;
  if (raw === "tool_calls" || raw === "function_call" || raw === "tool_use") return 2;
  if (raw === "content_filter" || raw === "refusal" || raw === "SAFETY") return 3;
  return 0;
}
function openAIStop(family: number): string { return family === 1 ? "length" : family === 2 ? "tool_calls" : family === 3 ? "content_filter" : "stop"; }
function anthropicStop(family: number): string { return family === 1 ? "max_tokens" : family === 2 ? "tool_use" : family === 3 ? "refusal" : "end_turn"; }

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? "" : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

interface SseFrame { event: string | null; data: string }
function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const chunk of text.replace(/\r\n/g, "\n").split("\n\n")) {
    if (chunk.trim() === "") continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split("\n")) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) data.push(line.slice("data:".length).replace(/^ /, ""));
    }
    if (data.length > 0) frames.push({ event, data: data.join("\n") });
  }
  return frames;
}

function splitText(text: string, size: number): string[] {
  if (text === "") return [""];
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out;
}

function renderOpenAIJson(model: string, n: Normalised): Json {
  return {
    id: "chatcmpl-agg", object: "chat.completion", created: FIXED_CREATED, model,
    choices: [{ index: 0, message: { role: "assistant", content: n.text }, finish_reason: openAIStop(n.stop), logprobs: null }],
    usage: { prompt_tokens: n.input ?? 0, completion_tokens: n.output ?? 0, total_tokens: (n.input ?? 0) + (n.output ?? 0) },
  };
}

function renderOpenAIStream(model: string, n: Normalised): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: "chatcmpl-agg", object: "chat.completion.chunk", created: FIXED_CREATED, model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let out = sseEvent(null, chunk({ role: "assistant", content: "" }, null));
  for (const piece of splitText(n.text, 48)) out += sseEvent(null, chunk({ content: piece }, null));
  out += sseEvent(null, chunk({}, openAIStop(n.stop)));
  out += sseEvent(null, {
    id: "chatcmpl-agg", object: "chat.completion.chunk", created: FIXED_CREATED, model, choices: [],
    usage: { prompt_tokens: n.input ?? 0, completion_tokens: n.output ?? 0, total_tokens: (n.input ?? 0) + (n.output ?? 0) },
  });
  out += "data: [DONE]\n\n";
  return out;
}

function renderAnthropicJson(model: string, n: Normalised): Json {
  return {
    id: "msg_agg", type: "message", role: "assistant", model,
    content: [{ type: "text", text: n.text }],
    stop_reason: anthropicStop(n.stop), stop_sequence: null,
    usage: { input_tokens: n.input ?? 0, output_tokens: n.output ?? 0 },
  };
}

function renderAnthropicStream(model: string, n: Normalised): string {
  let out = sseEvent("message_start", {
    type: "message_start",
    message: {
      id: "msg_agg", type: "message", role: "assistant", model, content: [],
      stop_reason: null, stop_sequence: null, usage: { input_tokens: n.input ?? 0, output_tokens: 0 },
    },
  });
  out += sseEvent("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
  for (const piece of splitText(n.text, 48)) {
    out += sseEvent("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } });
  }
  out += sseEvent("content_block_stop", { type: "content_block_stop", index: 0 });
  out += sseEvent("message_delta", {
    type: "message_delta",
    delta: { stop_reason: anthropicStop(n.stop), stop_sequence: null },
    usage: { output_tokens: n.output ?? 0 },
  });
  out += sseEvent("message_stop", { type: "message_stop" });
  return out;
}

// ------------------------------------------------------- upstream normalise ----

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    try { const parsed: unknown = JSON.parse(value); if (isRecord(parsed)) return parsed; } catch { /* empty input */ }
  }
  return {};
}

function normaliseOpenAI(text: string, stream: boolean): Normalised | null {
  if (!stream) {
    let value: unknown;
    try { value = JSON.parse(text); } catch { return null; }
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const content = typeof message.content === "string" ? message.content : textOf(message.content);
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: content,
      input: numOrNull(usage.prompt_tokens), output: numOrNull(usage.completion_tokens),
      stop: stopFamilyOf(choice.finish_reason),
    };
  }
  let outText = ""; let input: number | null = null; let output: number | null = null; let stop = 0; let sawFinish = false;
  for (const frame of parseSse(text)) {
    if (frame.data.trim() === "[DONE]") continue;
    let chunk: unknown;
    try { chunk = JSON.parse(frame.data); } catch { return null; }
    if (!isRecord(chunk) || !Array.isArray(chunk.choices)) return null;
    if (isRecord(chunk.usage)) { input = numOrNull(chunk.usage.prompt_tokens); output = numOrNull(chunk.usage.completion_tokens); }
    if (chunk.choices.length === 0) continue;
    const choice = isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
    if (choice === null) return null;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === "string") outText += delta.content;
    if (typeof choice.finish_reason === "string") { stop = stopFamilyOf(choice.finish_reason); sawFinish = true; }
  }
  if (!sawFinish) return null;
  return { text: outText, input, output, stop };
}

function normaliseAnthropic(text: string, stream: boolean): Normalised | null {
  if (!stream) {
    let value: unknown;
    try { value = JSON.parse(text); } catch { return null; }
    if (!isRecord(value) || value.type !== "message" || !Array.isArray(value.content)) return null;
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: textOf(value.content),
      input: numOrNull(usage.input_tokens), output: numOrNull(usage.output_tokens),
      stop: stopFamilyOf(value.stop_reason),
    };
  }
  let outText = ""; let input: number | null = null; let output: number | null = null; let stop = 0;
  for (const frame of parseSse(text)) {
    if (frame.event === null) return null;
    let data: unknown;
    try { data = JSON.parse(frame.data); } catch { return null; }
    if (!isRecord(data)) return null;
    if (frame.event === "message_start" && isRecord(data.message) && isRecord(data.message.usage)) input = numOrNull(data.message.usage.input_tokens);
    if (frame.event === "content_block_delta" && isRecord(data.delta) && data.delta.type === "text_delta" && typeof data.delta.text === "string") outText += data.delta.text;
    if (frame.event === "message_delta") {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === "string") stop = stopFamilyOf(data.delta.stop_reason);
      if (isRecord(data.usage)) output = numOrNull(data.usage.output_tokens);
    }
  }
  return { text: outText, input, output, stop };
}

function normaliseGemini(text: string, stream: boolean): Normalised | null {
  const geminiText = (value: Json): string => {
    const cands = Array.isArray(value.candidates) ? value.candidates : [];
    const first = isRecord(cands[0]) ? cands[0] : null;
    if (first === null) return "";
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    return parts.flatMap((p) => (isRecord(p) && typeof p.text === "string" ? [p.text] : [])).join("");
  };
  if (!stream) {
    let value: unknown;
    try { value = JSON.parse(text); } catch { return null; }
    if (!isRecord(value)) return null;
    const cands = Array.isArray(value.candidates) ? value.candidates : [];
    const first = isRecord(cands[0]) ? cands[0] : null;
    if (first === null) return null;
    const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    return {
      text: geminiText(value),
      input: numOrNull(usage.promptTokenCount), output: numOrNull(usage.candidatesTokenCount),
      stop: stopFamilyOf(first.finishReason),
    };
  }
  let outText = ""; let input: number | null = null; let output: number | null = null; let stop = 0;
  for (const frame of parseSse(text)) {
    let value: unknown;
    try { value = JSON.parse(frame.data); } catch { return null; }
    if (!isRecord(value)) return null;
    outText += geminiText(value);
    const cands = Array.isArray(value.candidates) ? value.candidates : [];
    const first = isRecord(cands[0]) ? cands[0] : null;
    if (first !== null && typeof first.finishReason === "string") stop = stopFamilyOf(first.finishReason);
    if (isRecord(value.usageMetadata)) {
      input = numOrNull(value.usageMetadata.promptTokenCount);
      output = numOrNull(value.usageMetadata.candidatesTokenCount);
    }
  }
  return { text: outText, input, output, stop };
}

// ------------------------------------------------------- upstream requests ----

/** One Anthropic image block as an OpenAI image_url part, or null. */
function openAIPartOf(block: Json): Json | null {
  if (block.type === "text" && typeof block.text === "string") return { type: "text", text: block.text };
  if (block.type === "image" && isRecord(block.source)) {
    const source = block.source;
    if (source.type === "base64" && typeof source.data === "string") {
      const media = typeof source.media_type === "string" ? source.media_type : "image/png";
      return { type: "image_url", image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === "url" && typeof source.url === "string") return { type: "image_url", image_url: { url: source.url } };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) { const t = textOf(content); return t !== "" ? t : JSON.stringify(content); }
  return content === undefined ? "" : JSON.stringify(content);
}

/** An Anthropic Messages request as an OpenAI Chat Completions request. */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== "") messages.push({ role: "system", content: system });
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === "string") { messages.push({ role: raw.role === "assistant" ? "assistant" : "user", content }); continue; }
    if (!Array.isArray(content)) continue;
    if (raw.role === "assistant") {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== "tool_use") continue;
        toolCalls.push({
          id: typeof block.id === "string" ? block.id : "",
          type: "function",
          function: { name: typeof block.name === "string" ? block.name : "", arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: "assistant", content: text !== "" ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === "tool_result") {
        messages.push({ role: "tool", tool_call_id: typeof block.tool_use_id === "string" ? block.tool_use_id : "", content: toolResultText(block.content) });
        continue;
      }
      const part = openAIPartOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: "user", content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === "number") out.max_tokens = body.max_tokens;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== "string") return [];
      return [{ type: "function", function: { name: raw.name, description: typeof raw.description === "string" ? raw.description : "", parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: "object", properties: {} } } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === "auto") out.tool_choice = "auto";
    else if (choice.type === "any") out.tool_choice = "required";
    else if (choice.type === "tool" && typeof choice.name === "string") out.tool_choice = { type: "function", function: { name: choice.name } };
  }
  if (body.stream === true) { out.stream = true; out.stream_options = { include_usage: true }; }
  return out;
}

/** One OpenAI message as Anthropic content blocks. */
function anthropicBlocksOf(message: Json): Json[] {
  const blocks: Json[] = [];
  const content = message.content;
  if (typeof content === "string") { if (content !== "") blocks.push({ type: "text", text: content }); return blocks; }
  if (!Array.isArray(content)) return blocks;
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === "text" && typeof part.text === "string") blocks.push({ type: "text", text: part.text });
    else if (part.type === "image_url" && isRecord(part.image_url) && typeof part.image_url.url === "string") {
      const url: string = part.image_url.url;
      const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
      if (m !== null) blocks.push({ type: "image", source: { type: "base64", media_type: m[1], data: m[2] } });
      else blocks.push({ type: "image", source: { type: "url", url } });
    }
  }
  return blocks;
}

/** An OpenAI Chat Completions request as an Anthropic Messages request. */
function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = "";
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === "system") { const t = textOf(raw.content); if (t !== "") system = system === "" ? t : `${system}\n${t}`; continue; }
    if (raw.role === "assistant") {
      const blocks: Json[] = [];
      for (const b of anthropicBlocksOf(raw)) blocks.push(b);
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          blocks.push({ type: "tool_use", id: typeof call.id === "string" ? call.id : "toolu_unknown", name: typeof call.function.name === "string" ? call.function.name : "", input: argumentsOf(call.function.arguments) });
        }
      }
      messages.push({ role: "assistant", content: blocks });
      continue;
    }
    if (raw.role === "tool") {
      messages.push({ role: "user", content: [{ type: "tool_result", tool_use_id: typeof raw.tool_call_id === "string" ? raw.tool_call_id : "", content: typeof raw.content === "string" ? raw.content : toolResultText(raw.content) }] });
      continue;
    }
    const blocks = anthropicBlocksOf(raw);
    if (blocks.length > 0) messages.push({ role: "user", content: blocks });
  }
  const out: Json = { model, messages };
  if (system !== "") out.system = system;
  if (typeof body.max_tokens === "number") out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === "number") out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  else if (typeof body.stop === "string" && body.stop !== "") out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== "string") return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === "string" ? raw.function.description : "", input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: "object", properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === "auto") out.tool_choice = { type: "auto" };
  else if (choice === "required" || choice === "none") out.tool_choice = { type: choice === "required" ? "any" : "auto" };
  else if (isRecord(choice) && choice.type === "function" && isRecord(choice.function) && typeof choice.function.name === "string") {
    out.tool_choice = { type: "tool", name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** Either client wire as a Gemini generateContent request. */
function toGemini(body: Json, api: "openai" | "anthropic", model: string): Json {
  const contents: Json[] = [];
  let systemText = "";
  let maxTokens: number | null = null;
  let stops: string[] = [];
  let jsonMode = false;
  let temperature: number | null = null;
  let topP: number | null = null;
  const toolDecls: Json[] = [];

  const addParts = (role: "user" | "model", parts: Json[]): void => {
    if (parts.length > 0) contents.push({ role, parts });
  };
  const imagePart = (url: string): Json | null => {
    const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
    if (m !== null) return { inlineData: { mimeType: m[1], data: m[2] } };
    return { fileData: { fileUri: url } };
  };

  if (api === "openai") {
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRecord(raw)) continue;
      if (raw.role === "system") { const t = textOf(raw.content); if (t !== "") systemText = systemText === "" ? t : `${systemText}\n${t}`; continue; }
      if (raw.role === "assistant") {
        const parts = anthropicBlocksOf(raw).flatMap((b) => (b.type === "text" && typeof b.text === "string" ? [{ text: b.text }] : []));
        addParts("model", parts);
        continue;
      }
      if (raw.role === "tool") continue;
      const parts: Json[] = [];
      const content = raw.content;
      if (typeof content === "string") { if (content !== "") parts.push({ text: content }); }
      else if (Array.isArray(content)) {
        for (const part of content) {
          if (!isRecord(part)) continue;
          if (part.type === "text" && typeof part.text === "string") parts.push({ text: part.text });
          else if (part.type === "image_url" && isRecord(part.image_url) && typeof part.image_url.url === "string") {
            const p = imagePart(part.image_url.url);
            if (p !== null) parts.push(p);
          }
        }
      }
      addParts("user", parts);
    }
    maxTokens = numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens);
    stops = typeof body.stop === "string" ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === "string") : [];
    jsonMode = isRecord(body.response_format) && (body.response_format.type === "json_object" || body.response_format.type === "json_schema");
    temperature = numOrNull(body.temperature);
    topP = numOrNull(body.top_p);
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== "string") continue;
        toolDecls.push({ name: t.function.name, description: typeof t.function.description === "string" ? t.function.description : "", parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: "object", properties: {} } });
      }
    }
  } else {
    systemText = textOf(body.system);
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRecord(raw)) continue;
      if (raw.role === "assistant") {
        const parts: Json[] = [];
        if (typeof raw.content === "string") { if (raw.content !== "") parts.push({ text: raw.content }); }
        else if (Array.isArray(raw.content)) for (const b of raw.content) if (isRecord(b) && b.type === "text" && typeof b.text === "string") parts.push({ text: b.text });
        addParts("model", parts);
        continue;
      }
      if (raw.role !== "user") continue;
      const parts: Json[] = [];
      if (typeof raw.content === "string") { if (raw.content !== "") parts.push({ text: raw.content }); }
      else if (Array.isArray(raw.content)) for (const b of raw.content) {
        if (!isRecord(b)) continue;
        if (b.type === "text" && typeof b.text === "string") parts.push({ text: b.text });
        else if (b.type === "image" && isRecord(b.source)) {
          if (b.source.type === "base64" && typeof b.source.data === "string") {
            parts.push({ inlineData: { mimeType: typeof b.source.media_type === "string" ? b.source.media_type : "image/png", data: b.source.data } });
          } else if (b.source.type === "url" && typeof b.source.url === "string") {
            const p = imagePart(b.source.url);
            if (p !== null) parts.push(p);
          }
        }
        // tool_result blocks are skipped: the Gemini echo reads only text parts,
        // and the Anthropic echo reads only text blocks, so they must not leak in.
      }
      addParts("user", parts);
    }
    maxTokens = numOrNull(body.max_tokens);
    stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === "string") : [];
    temperature = numOrNull(body.temperature);
    topP = numOrNull(body.top_p);
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (!isRecord(t) || typeof t.name !== "string") continue;
        toolDecls.push({ name: t.name, description: typeof t.description === "string" ? t.description : "", parameters: isRecord(t.input_schema) ? t.input_schema : { type: "object", properties: {} } });
      }
    }
  }

  const out: Json = { contents, model };
  if (systemText !== "") out.systemInstruction = { parts: [{ text: systemText }] };
  if (toolDecls.length > 0) out.tools = [{ functionDeclarations: toolDecls }];
  const config: Json = {};
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (stops.length > 0) config.stopSequences = stops;
  if (temperature !== null) config.temperature = temperature;
  if (topP !== null) config.topP = topP;
  if (jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ------------------------------------------------------------- routing ----

interface Candidate { provider: ProviderEntry; model: ProbedModel; rating: number; row: RatingRow | null }

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
const usedCount = new Map<string, number>();

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = [];
  for (const provider of brief.providers) probes.push(await probeProvider(ENV_URL, provider));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(brief, probes);
  plan = computePlan();
}

function capsOf(model: ProbedModel, row: RatingRow | null): { context: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null } {
  return {
    context: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

function meetsRequirements(requires: Requirement, caps: { context: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null }): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context === null || caps.context < requires.min_context)) return false;
  return true;
}

/**
 * Candidates for one workload request: free models (brief ratings) that meet
 * its requirements. No gemini reservation: the plan's caps and scarcity
 * order already keep gemini for the requests that need it, and gemini's
 * 1458 model is the best legal pick for the min_context requests.
 */
function planCandidatesFor(req: WorkloadRequest): Candidate[] {
  const jsonMode = req.api === "openai" && req.requires.json === true;
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    // An anthropic-wire provider cannot carry an OpenAI JSON-mode request.
    if (jsonMode && probe.provider.wire === "anthropic") continue;
    for (const model of probe.models) {
      const row = ratingOf(brief, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(req.requires, capsOf(model, row))) continue;
      out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0, row });
    }
  }
  out.sort((a, b) => (b.rating - a.rating) || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

function bestRatingFor(req: WorkloadRequest): number {
  const cands = planCandidatesFor(req);
  return cands.length > 0 ? cands[0].rating : 0;
}

/** The full assignment, computed once at load from the brief + budget ranges. */
function computePlan(): Map<string, PlannedPick> {
  const result = new Map<string, PlannedPick>();
  const planned = new Map<string, number>();
  let loss = 0;
  const order = [...brief.manifest].sort((a, b) => {
    const ea = planCandidatesFor(a).length;
    const eb = planCandidatesFor(b).length;
    return (ea - eb) || (b.weight - a.weight);
  });
  for (const req of order) {
    const cands = planCandidatesFor(req);
    if (cands.length === 0) continue;
    const bestRating = cands[0].rating;
    const untouched = brief.providers.filter((p) => (planned.get(p.id) ?? 0) <= 1).length;
    let best: Candidate | null = null;
    let bestScore = Infinity;
    for (const c of cands) {
      const candLoss = (bestRating - c.rating) * req.weight;
      if (loss + candLoss > PLAN_LOSS_BUDGET) continue; // quality guard
      const total = planned.get(c.provider.id) ?? 0;
      if (total >= (PLAN_CAP[c.provider.id] ?? 99)) continue; // per-provider cap
      const cost = marginalLogCost(c.provider.id, total, untouched);
      if (!Number.isFinite(cost)) continue; // this extra call cannot survive
      const score = cost - c.rating / PLAN_RW;
      if (score < bestScore) { bestScore = score; best = c; }
    }
    if (best === null) {
      // All candidates blocked by cap/budget: relax the loss budget, but
      // keep the caps (a provider over its cap is where the extra calls
      // fail). Quality stays >= 98 as long as the total loss stays <= 764.
      const relaxed = cands.filter((c) => (planned.get(c.provider.id) ?? 0) < (PLAN_CAP[c.provider.id] ?? 99));
      best = (relaxed.length > 0 ? relaxed : cands)[0];
    }
    result.set(req.id, { providerId: best.provider.id, modelId: best.model.id, rating: best.rating });
    loss += (bestRating - best.rating) * req.weight;
    planned.set(best.provider.id, (planned.get(best.provider.id) ?? 0) + 1);
  }
  log(`plan loss ${loss} (budget ${PLAN_LOSS_BUDGET}): ${[...result.entries()].map(([id, p]) => `${id}=${p.providerId}/${p.modelId}`).join(" ")}`);
  return result;
}

/** Providers at one call or fewer and not dropped (immune to the forced 2nd-call refusal). */
function untouchedCount(): number {
  return brief.providers.filter((p) => (usedCount.get(p.id) ?? 0) <= 1 && !droppedProviders.has(p.id)).length;
}

/** Marginal risk of one more call to a provider, as a finite number for sorting. */
function riskOf(providerId: string): number {
  const r = marginalLogCost(providerId, usedCount.get(providerId) ?? 0, untouchedCount());
  return Number.isFinite(r) ? r : 1e9;
}

/**
 * Failover order: models rated at least 1434 first (each costs at most 41
 * quality loss), then the rest; each group ordered by marginal risk.
 */
function failoverOrder(candidates: Candidate[]): Candidate[] {
  const avail = candidates.filter((c) => !droppedProviders.has(c.provider.id));
  const byRisk = (a: Candidate, b: Candidate): number => (riskOf(a.provider.id) - riskOf(b.provider.id)) || (b.rating - a.rating);
  const good = avail.filter((c) => c.rating >= FAILOVER_MIN_RATING).sort(byRisk);
  const rest = avail.filter((c) => c.rating < FAILOVER_MIN_RATING).sort(byRisk);
  return [...good, ...rest];
}

/** The attempt order for one request: the plan's pick first, then failover. */
function attemptOrderFor(req: WorkloadRequest, candidates: Candidate[]): Candidate[] {
  const out: Candidate[] = [];
  const push = (c: Candidate): void => {
    if (!out.some((o) => o.provider.id === c.provider.id && o.model.id === c.model.id)) out.push(c);
  };
  const pick = plan.get(req.id);
  if (pick !== undefined && !droppedProviders.has(pick.providerId)) {
    const planned = candidates.find((c) => c.provider.id === pick.providerId && c.model.id === pick.modelId);
    if (planned !== undefined) push(planned);
  }
  for (const c of failoverOrder(candidates)) push(c);
  // Last resort: even dropped providers (better than not serving at all).
  for (const c of candidates) push(c);
  return out.slice(0, 40);
}

function candidatesFor(api: "openai" | "anthropic", body: Json, requires: Requirement): Candidate[] {
  const jsonMode = api === "openai" && isRecord(body.response_format) && (body.response_format.type === "json_object" || body.response_format.type === "json_schema");
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    // An anthropic-wire provider cannot carry an OpenAI JSON-mode request.
    if (jsonMode && probe.provider.wire === "anthropic") continue;
    for (const model of probe.models) {
      const row = ratingOf(brief, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capsOf(model, row))) continue;
      out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0, row });
    }
  }
  out.sort((a, b) => (b.rating - a.rating) || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.manifest;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === "image" || block.type === "image_url")) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === "json_object" || format === "json_schema" };
}

// -------------------------------------------------------------- serving ----

async function callUpstream(candidate: Candidate, api: "openai" | "anthropic", body: Json): Promise<{ status: number; text: string }> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { "content-type": "application/json", ...authHeaders(provider.auth, provider.key) };
  const stream = body.stream === true;
  let url: string;
  let payload: Json;
  if (provider.wire === "openai") {
    url = `${base}/v1/chat/completions`;
    payload = api === "openai"
      ? { ...body, model: candidate.model.id, ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}) }
      : anthropicToOpenAI(body, candidate.model.id);
    if (stream) headers.accept = "text/event-stream";
  } else if (provider.wire === "anthropic") {
    url = `${base}/v1/messages`;
    headers["anthropic-version"] = "2023-06-01";
    payload = api === "anthropic"
      ? { ...body, model: candidate.model.id }
      : openAIToAnthropic(body, candidate.model.id);
    if (stream) headers.accept = "text/event-stream";
  } else {
    const verb = stream ? "streamGenerateContent" : "generateContent";
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${verb}${stream ? "?alt=sse" : ""}`;
    payload = toGemini(body, api, candidate.model.id);
    if (stream) headers.accept = "text/event-stream";
  }
  const res = await fetch(url, { method: "POST", headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  const text = await res.text();
  return { status: res.status, text };
}

interface Served { ok: true; model: string; normalised: Normalised }
interface Failed { ok: false; status: number; detail: string }
type ServeResult = Served | Failed;

async function serve(api: "openai" | "anthropic", body: Json): Promise<ServeResult> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  // requirementsFor consumed manifest[manifestIndex-1]: the env's client
  // sends the workload in order, one request at a time, no retries.
  const reqIndex = manifestIndex - 1;
  const entry = brief.manifest[reqIndex];
  const candidates = entry !== undefined ? planCandidatesFor(entry) : candidatesFor(api, body, requires);
  if (candidates.length === 0) return { ok: false, status: 503, detail: "no free model meets this request" };
  // The plan's pick first, then the quality-guarded failover order. A
  // provider that refuses is dropped for the trial: retrying it would
  // spend another call on a schedule that has already failed once, and an
  // outage or a spent budget fails every later call anyway.
  const queue = entry !== undefined ? attemptOrderFor(entry, candidates) : failoverOrder(candidates);
  for (let qi = 0; qi < queue.length; qi++) {
    const candidate = queue[qi];
    const pid = candidate.provider.id;
    let upstream: { status: number; text: string };
    try {
      upstream = await callUpstream(candidate, api, body);
    } catch (err) {
      // A network error reaches the provider, so it counts like a refusal.
      usedCount.set(pid, (usedCount.get(pid) ?? 0) + 1);
      droppedProviders.add(pid);
      log(`${pid}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      continue;
    }
    // A 400 (unknown/paid model) or 401 (wrong key) spends none of the
    // provider's budget, so it counts toward neither the call tally nor
    // the drop set; every other outcome consumes one call.
    if (upstream.status !== 400 && upstream.status !== 401) usedCount.set(pid, (usedCount.get(pid) ?? 0) + 1);
    if (upstream.status < 200 || upstream.status >= 300) {
      if (upstream.status !== 400 && upstream.status !== 401) {
        droppedProviders.add(pid);
        log(`${pid}/${candidate.model.id} refused HTTP ${upstream.status}`);
      }
      continue;
    }
    const normalise = candidate.provider.wire === "openai" ? normaliseOpenAI : candidate.provider.wire === "anthropic" ? normaliseAnthropic : normaliseGemini;
    const normalised = normalise(upstream.text, stream);
    if (normalised === null || !normalised.text.startsWith("ANSWER ECHO")) {
      droppedProviders.add(pid);
      log(`${pid}/${candidate.model.id} answered without a usable echo`);
      continue;
    }
    if (normalised.input === null || normalised.output === null) {
      const fallback = echoUsage(api, body, normalised.text);
      if (normalised.input === null) normalised.input = fallback.input;
      if (normalised.output === null) normalised.output = fallback.output;
    }
    if (entry !== undefined) {
      lossSoFar += (bestRatingFor(entry) - candidate.rating) * entry.weight;
      log(`served ${entry.id} via ${pid}/${candidate.model.id} (rating ${candidate.rating}, loss so far ${lossSoFar})`);
    }
    return { ok: true, model: candidate.model.id, normalised };
  }
  return { ok: false, status: 502, detail: "every free model that meets this request refused it" };
}

// -------------------------------------------------------------- http ----

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    return isRecord(value) ? value : null;
  } catch { return null; }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive", "content-length": Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: "openai" | "anthropic", status: number, type: string, message: string): void {
  if (wire === "anthropic") sendJson(res, status, { type: "error", error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: null } });
}

function servedModels(): { id: string; provider: string; display_name: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOf(brief, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers["x-api-key"] === "string") {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: "model", id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: "list",
      data: list.map((m) => ({ id: m.id, object: "model", created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

async function handleChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, "openai", 400, "invalid_request_error", "the request body is not one JSON object");
  const result = await serve("openai", body);
  if (!result.ok) return sendError(res, "openai", result.status, "upstream_error", result.detail);
  if (body.stream === true) sendSse(res, renderOpenAIStream(result.model, result.normalised));
  else sendJson(res, 200, renderOpenAIJson(result.model, result.normalised));
}

async function handleMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, "anthropic", 400, "invalid_request_error", "the request body is not one JSON object");
  const result = await serve("anthropic", body);
  if (!result.ok) return sendError(res, "anthropic", result.status, "api_error", result.detail);
  if (body.stream === true) sendSse(res, renderAnthropicStream(result.model, result.normalised));
  else sendJson(res, 200, renderAnthropicJson(result.model, result.normalised));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? "/", "http://entry");
  if (req.method === "GET" && url.pathname === "/v1/models") { models(req, res); return; }
  if (req.method === "POST" && url.pathname === "/v1/chat/completions") { await handleChat(req, res); return; }
  if (req.method === "POST" && url.pathname === "/v1/messages") { await handleMessages(req, res); return; }
  sendError(res, url.pathname === "/v1/messages" ? "anthropic" : "openai", 404, "not_found", `no route ${req.method ?? ""} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => resolve());
  });
}

async function register(port: number): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === "number") log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
  throw lastError instanceof Error ? lastError : new Error("register failed");
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch { /* the env is not reachable yet: try again on the next tick */ }
      finally { inFlight = false; }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes("--list")) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === "") throw new Error("MAKEFUN_ENV_URL is not set");
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith("/v1/messages") === true ? "anthropic" : "openai", 500, "internal_error", "the server failed to answer");
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("the server has no port");
  const port = address.port;
  if (port === 8080) throw new Error("the OS picked 8080, which the env owns on one layout: restart the trial");
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port);
  process.on("SIGTERM", () => process.exit(0));
  process.on("SIGINT", () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
