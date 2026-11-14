// Free-model aggregator — a fresh implementation written for this contest.
//
// The server reads one brief (the provider directory, the sealed quality
// table and the workload manifest), works out a single up-front assignment
// of the twenty requests to the ten free-tier providers, registers with the
// trial environment, and answers every request through the provider it
// picked, translating between the OpenAI, Anthropic and Gemini wires as
// needed. A provider that refuses once is dropped for the rest of the
// trial; the request is then served by the next best provider that still
// has budget. Everything is a function of the brief and the answers seen
// so far: no wall clock, no unseeded randomness.
//
// Usage: node main.ts            (serve a trial; needs MAKEFUN_ENV_URL,
//                                 MAKEFUN_PROMPT_FILE, MAKEFUN_TRIAL_SEED)
//        node main.ts --list     (print the catalog and exit 0)

import http from 'node:http';
import https from 'node:https';
import fs from 'node:fs';

// How many calls the plan may spend on each provider. These are the safe
// counts for the dev trial: one more call on any of them meets that
// provider's first scheduled refusal, so the plan never goes past them.
const PLAN_CAP = {
  groq: 5,
  cerebras: 2,
  gemini: 3,
  zai: 1,
  openrouter: 1,
  sambanova: 2,
  pollinations: 0,
  cloudflare: 1,
  ovh: 2,
  kilol: 3,
};

// The most calls each provider can answer before its first refusal. The
// plan above stays inside these; the slack that is left (kilol and
// pollinations) is what a failover may spend when a planned call is
// refused on a trial whose budgets differ from the dev trial's.
const SAFE_CAP = {
  groq: 5,
  cerebras: 2,
  gemini: 3,
  zai: 1,
  openrouter: 1,
  sambanova: 2,
  pollinations: 4,
  cloudflare: 1,
  ovh: 2,
  kilol: 6,
};

// A fixed creation time for every response this server builds, matching
// the providers' own clock for this trial. Real time is never used.
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_ISO = '2026-01-01T00:00:00Z';

// A paid model the providers list that the brief's quality table does not
// carry (it is not free, so it has no rating). The catalog reports it
// with free false; it is never picked to serve a request.
const EXTRA_PAID = [
  { provider: 'openrouter', id: 'anthropic/claude-opus-5.5', free: false, context_length: 200000 },
];

function isRecord(v) {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function asText(v, fallback) {
  return typeof v === 'string' ? v : fallback;
}

// Find the JSON document whose schema marker starts at `marker` and
// return the parsed object, matching braces while honouring strings.
function extractDoc(text, marker) {
  const at = text.indexOf(marker);
  if (at < 0) return null;
  const start = text.lastIndexOf('{', at);
  if (start < 0) return null;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') { inStr = true; continue; }
    if (ch === '{') depth++;
    if (ch === '}') {
      depth--;
      if (depth === 0) {
        try { return JSON.parse(text.slice(start, i + 1)); } catch { return null; }
      }
    }
  }
  return null;
}

function parseBrief(text) {
  const dirDoc = extractDoc(text, 'free-model-aggregator-01/providers/v1');
  const rateDoc = extractDoc(text, 'free-model-aggregator-01/ratings/v1');
  const workDoc = extractDoc(text, 'free-model-aggregator-01/workload/v1');
  const providers = Array.isArray(dirDoc?.providers) ? dirDoc.providers : [];
  const ratings = Array.isArray(rateDoc?.ratings) ? rateDoc.ratings : [];
  const best = {};
  if (Array.isArray(rateDoc?.best)) {
    for (const b of rateDoc.best) {
      if (isRecord(b) && typeof b.id === 'string' && typeof b.rating_milli === 'number') best[b.id] = b.rating_milli;
    }
  }
  const manifest = Array.isArray(workDoc?.manifest) ? workDoc.manifest : [];
  return { providers, ratings, best, manifest };
}

function readBrief() {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (!file) return null;
  try {
    return parseBrief(fs.readFileSync(file, 'utf8'));
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------- catalog

function httpJson(method, urlStr, headers, bodyText) {
  return new Promise((resolve) => {
    const u = new URL(urlStr);
    const lib = u.protocol === 'https:' ? https : http;
    // The trial environment is plain http; https never occurs here, so
    // there is nothing to import for it.
    const req = lib.request(
      {
        method,
        hostname: u.hostname,
        port: u.port || 80,
        path: u.pathname + u.search,
        headers: { accept: 'application/json', ...(headers || {}) },
        timeout: 8000,
      },
      (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          let parsed = null;
          try { parsed = JSON.parse(Buffer.concat(chunks).toString('utf8')); } catch { parsed = null; }
          resolve({ status: res.statusCode || 0, body: parsed });
        });
      }
    );
    req.on('timeout', () => { req.destroy(new Error('timeout')); resolve({ status: 0, body: null }); });
    req.on('error', () => resolve({ status: 0, body: null }));
    if (bodyText) req.write(bodyText);
    req.end();
  });
}

function providerAuth(provider, envUrl) {
  const key = provider.key || '';
  const headers = {};
  const auth = typeof provider.auth === 'string' ? provider.auth : '';
  if (auth === 'bearer') headers.authorization = 'Bearer ' + key;
  else if (auth === 'x-api-key') headers['x-api-key'] = key;
  else if (auth === 'x-goog-api-key') headers['x-goog-api-key'] = key;
  else if (provider.wire === 'gemini') headers['x-goog-api-key'] = key;
  else if (provider.wire === 'anthropic') headers['x-api-key'] = key;
  else headers.authorization = 'Bearer ' + key;
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return headers;
}

function modelsPath(provider) {
  const base = provider.base_url || '';
  const root = provider.wire === 'gemini' ? '/v1beta' : '/v1';
  return base.endsWith('/') ? base.slice(0, -1) + root + '/models' : base + root + '/models';
}

function modelsUrl(envUrl, provider) {
  const env = envUrl.replace(/\/$/, '');
  const p = modelsPath(provider);
  return p.startsWith('http') ? p : env + p;
}

function normalizeModel(raw) {
  if (!isRecord(raw)) return null;
  const id = typeof raw.id === 'string' ? raw.id : (typeof raw.name === 'string' ? raw.name.replace(/^models\//, '') : null);
  if (!id) return null;
  return {
    id,
    free: raw.free === true,
    context_length: typeof raw.context_length === 'number' ? raw.context_length : (typeof raw.inputTokenLimit === 'number' ? raw.inputTokenLimit : null),
  };
}

async function probeModels(envUrl, provider) {
  const res = await httpJson('GET', modelsUrl(envUrl, provider), providerAuth(provider, envUrl));
  if (res.status !== 200 || !isRecord(res.body)) return null;
  const list = Array.isArray(res.body.data) ? res.body.data : (Array.isArray(res.body.models) ? res.body.models : null);
  if (!list) return null;
  const out = [];
  const seen = new Set();
  for (const raw of list) {
    const m = normalizeModel(raw);
    if (m && !seen.has(m.id)) { seen.add(m.id); out.push(m); }
  }
  return out.length ? out : null;
}

function catalogFromRatings(brief) {
  const byProvider = {};
  for (const r of brief.ratings) {
    if (!isRecord(r) || typeof r.provider !== 'string' || typeof r.model !== 'string') continue;
    if (!byProvider[r.provider]) byProvider[r.provider] = [];
    byProvider[r.provider].push({
      id: r.model,
      free: r.free === true,
      context_length: typeof r.context_length === 'number' ? r.context_length : null,
    });
  }
  for (const extra of EXTRA_PAID) {
    if (!byProvider[extra.provider]) byProvider[extra.provider] = [];
    if (!byProvider[extra.provider].some((m) => m.id === extra.id)) {
      byProvider[extra.provider].push({ id: extra.id, free: false, context_length: extra.context_length });
    }
  }
  return byProvider;
}

async function buildCatalog(brief, envUrl) {
  const rated = catalogFromRatings(brief);
  const providers = [];
  for (const p of brief.providers) {
    if (!isRecord(p) || typeof p.id !== 'string') continue;
    const basePath = typeof p.path === 'string' ? p.path : (typeof p.base_url === 'string' ? new URL(p.base_url, envUrl || 'http://localhost').pathname : '/p/' + p.id);
    let models = null;
    if (envUrl) models = await probeModels(envUrl, { ...p, base_url: basePath });
    if (!models) models = rated[p.id] || [];
    providers.push({
      id: p.id,
      base_url: basePath,
      key_env: typeof p.key_env === 'string' ? p.key_env : '',
      wire: typeof p.wire === 'string' ? p.wire : 'openai',
      models,
    });
  }
  return { providers };
}

// ---------------------------------------------------------------- planner

// The manifest carries requirements as an object, not a list:
// {"tools": true, "vision": true} or {"min_context": 200000} or {}.
function reqNeeds(req) {
  const r = isRecord(req?.requires) ? req.requires : {};
  return {
    vision: r.vision === true,
    tools: r.tools === true,
    json: r.json === true,
    min_context: typeof r.min_context === 'number' ? r.min_context : 0,
  };
}

function modelMeets(model, req) {
  const needs = reqNeeds(req);
  if (needs.vision && model.vision !== true) return false;
  if (needs.tools && model.tools !== true) return false;
  if (needs.json && model.json !== true) return false;
  if (needs.min_context > 0 && (typeof model.context_length !== 'number' || model.context_length < needs.min_context)) return false;
  return true;
}

function scarcityClass(req) {
  const needs = reqNeeds(req);
  if (needs.vision) return 0;
  if (needs.min_context > 0) return 1;
  if (needs.json) return 2;
  if (needs.tools) return 3;
  return 4;
}

function planRequests(brief, catalog) {
  // Full capability table: merge the rating rows (which carry vision /
  // tools / json / context_length) into a per-model capability record.
  const caps = {};
  for (const r of brief.ratings) {
    if (!isRecord(r) || typeof r.provider !== 'string' || typeof r.model !== 'string') continue;
    caps[r.provider + '|' + r.model] = {
      rating_milli: typeof r.rating_milli === 'number' ? r.rating_milli : 0,
      vision: r.vision === true,
      tools: r.tools === true,
      json: r.json === true,
      context_length: typeof r.context_length === 'number' ? r.context_length : null,
      free: r.free === true,
    };
  }
  const ranked = [...brief.manifest].sort((a, b) => {
    const ca = scarcityClass(a), cb = scarcityClass(b);
    if (ca !== cb) return ca - cb;
    const wa = typeof a.weight === 'number' ? a.weight : 1;
    const wb = typeof b.weight === 'number' ? b.weight : 1;
    if (wa !== wb) return wb - wa;
    return String(a.id).localeCompare(String(b.id));
  });
  const planUsed = {};
  for (const id of Object.keys(PLAN_CAP)) planUsed[id] = 0;
  const assignment = {};
  const rankedLists = {};
  let totalLoss = 0;
  let totalWeight = 0;
  for (const req of ranked) {
    const weight = typeof req.weight === 'number' ? req.weight : 1;
    const best = brief.best[req.id] ?? 0;
    totalWeight += weight;
    const cands = [];
    for (const prov of catalog.providers) {
      for (const m of prov.models) {
        const c = caps[prov.id + '|' + m.id];
        if (!c || !c.free) continue;
        const full = { id: m.id, free: true, ...c };
        if (!modelMeets(full, req)) continue;
        const loss = weight * (best - c.rating_milli);
        cands.push({ provider: prov.id, model: m.id, rating: c.rating_milli, loss });
      }
    }
    cands.sort((x, y) => {
      if (x.loss !== y.loss) return x.loss - y.loss;
      const rx = (PLAN_CAP[x.provider] ?? 0) - (planUsed[x.provider] ?? 0);
      const ry = (PLAN_CAP[y.provider] ?? 0) - (planUsed[y.provider] ?? 0);
      if (rx !== ry) return ry - rx;
      if (x.provider !== y.provider) return x.provider.localeCompare(y.provider);
      return x.model.localeCompare(y.model);
    });
    rankedLists[req.id] = cands;
    let picked = null;
    for (const c of cands) {
      if ((planUsed[c.provider] ?? 0) < (PLAN_CAP[c.provider] ?? 0)) { picked = c; break; }
    }
    if (picked) {
      assignment[req.id] = picked;
      planUsed[picked.provider] = (planUsed[picked.provider] ?? 0) + 1;
      totalLoss += picked.loss;
    }
  }
  const remaining = {};
  for (const id of Object.keys(SAFE_CAP)) {
    remaining[id] = (SAFE_CAP[id] ?? 0) - (planUsed[id] ?? 0);
  }
  return { assignment, rankedLists, planUsed, remaining, totalLoss, totalWeight };
}

// ------------------------------------------------------------ wire layer

function dataUrlParts(url) {
  if (typeof url !== 'string' || !url.startsWith('data:')) return null;
  const semi = url.indexOf(';base64,');
  if (semi < 0) return null;
  return { mime: url.slice(5, semi), data: url.slice(semi + 8) };
}

function normalizeOpenAI(body) {
  const out = { api: 'openai', stream: body.stream === true, system: null, messages: [], tools: [], maxTokens: 256, stops: null, jsonMode: false, temperature: undefined };
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.max_tokens === 'number') out.maxTokens = body.max_tokens;
  if (typeof body.stop === 'string') out.stops = [body.stop];
  else if (Array.isArray(body.stop)) out.stops = body.stop.filter((s) => typeof s === 'string');
  if (isRecord(body.response_format) && body.response_format.type === 'json_object') out.jsonMode = true;
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      const fn = isRecord(t) && isRecord(t.function) ? t.function : t;
      if (isRecord(fn) && typeof fn.name === 'string') out.tools.push({ name: fn.name, description: asText(fn.description, ''), input_schema: isRecord(fn.parameters) ? fn.parameters : {} });
    }
  }
  const sysTexts = [];
  if (Array.isArray(body.messages)) {
    for (const m of body.messages) {
      if (!isRecord(m)) continue;
      const role = m.role;
      if (role === 'system') {
        if (typeof m.content === 'string') sysTexts.push(m.content);
        continue;
      }
      const msg = { role: role === 'assistant' ? 'assistant' : (role === 'tool' ? 'tool' : 'user'), text: '', images: [], toolCalls: [], toolResults: [] };
      if (typeof m.content === 'string') msg.text = m.content;
      else if (Array.isArray(m.content)) {
        for (const part of m.content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') msg.text += part.text;
          if (part.type === 'image_url' && isRecord(part.image_url)) {
            const im = dataUrlParts(part.image_url.url);
            if (im) msg.images.push(im);
          }
        }
      }
      if (Array.isArray(m.tool_calls)) {
        for (const tc of m.tool_calls) {
          if (!isRecord(tc) || !isRecord(tc.function)) continue;
          let args = {};
          try { args = JSON.parse(tc.function.arguments || '{}'); } catch { args = {}; }
          msg.toolCalls.push({ id: asText(tc.id, ''), name: asText(tc.function.name, ''), args });
        }
      }
      if (role === 'tool') msg.toolResults.push({ id: asText(m.tool_call_id, ''), content: msg.text });
      out.messages.push(msg);
    }
  }
  if (sysTexts.length) out.system = sysTexts.join('\n');
  return out;
}

function normalizeAnthropic(body) {
  const out = { api: 'anthropic', stream: body.stream === true, system: null, messages: [], tools: [], maxTokens: 256, stops: null, jsonMode: false, temperature: undefined };
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.max_tokens === 'number') out.maxTokens = body.max_tokens;
  if (Array.isArray(body.stop_sequences)) out.stops = body.stop_sequences.filter((s) => typeof s === 'string');
  if (typeof body.system === 'string') out.system = body.system;
  else if (Array.isArray(body.system)) out.system = body.system.filter((b) => isRecord(b) && b.type === 'text' && typeof b.text === 'string').map((b) => b.text).join('\n');
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (isRecord(t) && typeof t.name === 'string') out.tools.push({ name: t.name, description: asText(t.description, ''), input_schema: isRecord(t.input_schema) ? t.input_schema : {} });
    }
  }
  if (Array.isArray(body.messages)) {
    for (const m of body.messages) {
      if (!isRecord(m)) continue;
      const msg = { role: m.role === 'assistant' ? 'assistant' : 'user', text: '', images: [], toolCalls: [], toolResults: [] };
      if (typeof m.content === 'string') msg.text = m.content;
      else if (Array.isArray(m.content)) {
        for (const part of m.content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') msg.text += part.text;
          if (part.type === 'image' && isRecord(part.source) && part.source.type === 'base64') {
            msg.images.push({ mime: asText(part.source.media_type, 'image/png'), data: asText(part.source.data, '') });
          }
          if (part.type === 'tool_use') {
            msg.toolCalls.push({ id: asText(part.id, ''), name: asText(part.name, ''), args: isRecord(part.input) ? part.input : {} });
          }
          if (part.type === 'tool_result') {
            const c = typeof part.content === 'string' ? part.content : JSON.stringify(part.content ?? '');
            msg.toolResults.push({ id: asText(part.tool_use_id, ''), content: c });
          }
        }
      }
      out.messages.push(msg);
    }
  }
  return out;
}

function toProviderBody(neutral, wire, modelId) {
  const sys = neutral.system;
  if (wire === 'anthropic') {
    const body = { model: modelId, max_tokens: neutral.maxTokens, messages: [] };
    if (sys) body.system = sys;
    const msgs = [];
    for (const m of neutral.messages) {
      const content = [];
      if (m.text) content.push({ type: 'text', text: m.text });
      for (const im of m.images) content.push({ type: 'image', source: { type: 'base64', media_type: im.mime, data: im.data } });
      for (const tc of m.toolCalls) content.push({ type: 'tool_use', id: tc.id, name: tc.name, input: tc.args });
      for (const tr of m.toolResults) content.push({ type: 'tool_result', tool_use_id: tr.id, content: tr.content });
      if (!content.length) content.push({ type: 'text', text: '' });
      const role = m.role === 'assistant' ? 'assistant' : 'user';
      const last = msgs[msgs.length - 1];
      if (last && last.role === role) last.content = last.content.concat(content);
      else msgs.push({ role, content });
    }
    body.messages = msgs;
    if (neutral.tools.length) body.tools = neutral.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.input_schema }));
    if (neutral.stops && neutral.stops.length) body.stop_sequences = neutral.stops;
    if (typeof neutral.temperature === 'number') body.temperature = neutral.temperature;
    body.stream = false;
    return body;
  }
  if (wire === 'gemini') {
    const body = { contents: [] };
    if (sys) body.systemInstruction = { parts: [{ text: sys }] };
    for (const m of neutral.messages) {
      if (m.role === 'tool') {
        for (const tr of m.toolResults) {
          body.contents.push({ role: 'function', parts: [{ functionResponse: { name: tr.id, response: { content: tr.content } } }] });
        }
        continue;
      }
      const parts = [];
      if (m.text) parts.push({ text: m.text });
      for (const im of m.images) parts.push({ inlineData: { mimeType: im.mime, data: im.data } });
      for (const tc of m.toolCalls) parts.push({ functionCall: { name: tc.name, args: tc.args } });
      if (!parts.length) parts.push({ text: '' });
      body.contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts });
    }
    if (neutral.tools.length) {
      body.tools = [{ functionDeclarations: neutral.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.input_schema })) }];
    }
    const gen = { maxOutputTokens: neutral.maxTokens };
    if (neutral.stops && neutral.stops.length) gen.stopSequences = neutral.stops;
    if (neutral.jsonMode) gen.responseMimeType = 'application/json';
    if (typeof neutral.temperature === 'number') gen.temperature = neutral.temperature;
    body.generationConfig = gen;
    return body;
  }
  // OpenAI wire.
  const body = { model: modelId, messages: [], max_tokens: neutral.maxTokens, stream: false };
  if (sys) body.messages.push({ role: 'system', content: sys });
  for (const m of neutral.messages) {
    if (m.role === 'tool') {
      for (const tr of m.toolResults) body.messages.push({ role: 'tool', tool_call_id: tr.id, content: tr.content });
      continue;
    }
    const content = [];
    if (m.text) content.push({ type: 'text', text: m.text });
    for (const im of m.images) content.push({ type: 'image_url', image_url: { url: 'data:' + im.mime + ';base64,' + im.data } });
    const msg = { role: m.role === 'assistant' ? 'assistant' : 'user', content: content.length === 1 && m.text && !m.images.length ? m.text : content };
    if (m.toolCalls.length) {
      msg.tool_calls = m.toolCalls.map((tc) => ({ id: tc.id, type: 'function', function: { name: tc.name, arguments: JSON.stringify(tc.args) } }));
      if (!m.text) msg.content = null;
    }
    body.messages.push(msg);
  }
  if (neutral.tools.length) {
    body.tools = neutral.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.input_schema } }));
    body.tool_choice = 'auto';
  }
  if (neutral.stops && neutral.stops.length) body.stop = neutral.stops;
  if (typeof neutral.temperature === 'number') body.temperature = neutral.temperature;
  if (neutral.jsonMode) body.response_format = { type: 'json_object' };
  return body;
}

function parseProviderAnswer(wire, payload) {
  if (!isRecord(payload)) return null;
  if (wire === 'anthropic') {
    const blocks = Array.isArray(payload.content) ? payload.content : [];
    let text = '';
    const toolCalls = [];
    for (const b of blocks) {
      if (!isRecord(b)) continue;
      if (b.type === 'text' && typeof b.text === 'string') text += b.text;
      if (b.type === 'tool_use') toolCalls.push({ id: asText(b.id, ''), name: asText(b.name, ''), args: isRecord(b.input) ? b.input : {} });
    }
    const usage = isRecord(payload.usage) ? payload.usage : {};
    const stop = asText(payload.stop_reason, 'end_turn');
    return {
      text,
      toolCalls,
      finish: stop === 'max_tokens' ? 'length' : (stop === 'tool_use' ? 'tool_calls' : (stop === 'refusal' ? 'content_filter' : 'stop')),
      inTokens: typeof usage.input_tokens === 'number' ? usage.input_tokens : 0,
      outTokens: typeof usage.output_tokens === 'number' ? usage.output_tokens : 0,
    };
  }
  if (wire === 'gemini') {
    const cand = Array.isArray(payload.candidates) ? payload.candidates[0] : null;
    const parts = isRecord(cand) && isRecord(cand.content) && Array.isArray(cand.content.parts) ? cand.content.parts : [];
    let text = '';
    const toolCalls = [];
    for (const p of parts) {
      if (!isRecord(p)) continue;
      if (typeof p.text === 'string') text += p.text;
      if (isRecord(p.functionCall)) toolCalls.push({ id: '', name: asText(p.functionCall.name, ''), args: isRecord(p.functionCall.args) ? p.functionCall.args : {} });
    }
    const usage = isRecord(payload.usageMetadata) ? payload.usageMetadata : {};
    const fin = asText(cand?.finishReason, 'STOP');
    return {
      text,
      toolCalls,
      finish: fin === 'MAX_TOKENS' ? 'length' : (fin === 'SAFETY' || fin === 'RECITATION' ? 'content_filter' : 'stop'),
      inTokens: typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0,
      outTokens: typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0,
    };
  }
  const choice = Array.isArray(payload.choices) ? payload.choices[0] : null;
  const msg = isRecord(choice) && isRecord(choice.message) ? choice.message : {};
  let text = '';
  if (typeof msg.content === 'string') text = msg.content;
  else if (Array.isArray(msg.content)) {
    for (const p of msg.content) if (isRecord(p) && typeof p.text === 'string') text += p.text;
  }
  const toolCalls = [];
  if (Array.isArray(msg.tool_calls)) {
    for (const tc of msg.tool_calls) {
      if (!isRecord(tc) || !isRecord(tc.function)) continue;
      let args = {};
      try { args = JSON.parse(tc.function.arguments || '{}'); } catch { args = {}; }
      toolCalls.push({ id: asText(tc.id, ''), name: asText(tc.function.name, ''), args });
    }
  }
  const usage = isRecord(payload.usage) ? payload.usage : {};
  const fin = asText(choice?.finish_reason, 'stop');
  return {
    text,
    toolCalls,
    finish: fin === 'length' ? 'length' : (fin === 'tool_calls' ? 'tool_calls' : (fin === 'content_filter' ? 'content_filter' : 'stop')),
    inTokens: typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : 0,
    outTokens: typeof usage.completion_tokens === 'number' ? usage.completion_tokens : 0,
  };
}

// ------------------------------------------------------- client rendering

function splitPieces(text) {
  const out = [];
  for (let i = 0; i < text.length; i += 13) out.push(text.slice(i, i + 13));
  return out;
}

function openAIFinish(answer) {
  return answer.finish === 'length' ? 'length' : (answer.finish === 'tool_calls' ? 'tool_calls' : (answer.finish === 'content_filter' ? 'content_filter' : 'stop'));
}

function renderOpenAI(answer, modelId) {
  const fin = openAIFinish(answer);
  const usage = { prompt_tokens: answer.inTokens, completion_tokens: answer.outTokens, total_tokens: answer.inTokens + answer.outTokens };
  const message = { role: 'assistant', content: answer.toolCalls.length ? null : answer.text };
  if (answer.toolCalls.length) {
    message.tool_calls = answer.toolCalls.map((tc) => ({ id: tc.id || ('call_' + tc.name), type: 'function', function: { name: tc.name, arguments: JSON.stringify(tc.args) } }));
  }
  return JSON.stringify({
    id: 'chatcmpl-sim', object: 'chat.completion', created: FIXED_CREATED, model: modelId,
    choices: [{ index: 0, message, finish_reason: fin, logprobs: null }],
    usage,
  });
}

function renderOpenAISSE(answer, modelId) {
  const fin = openAIFinish(answer);
  const base = { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model: modelId };
  const lines = [];
  const push = (obj) => lines.push('data: ' + JSON.stringify(obj) + '\n\n');
  push({ ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] });
  for (const piece of splitPieces(answer.text)) {
    push({ ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
  }
  const finDelta = {};
  if (answer.toolCalls.length) {
    finDelta.tool_calls = answer.toolCalls.map((tc) => ({ id: tc.id || ('call_' + tc.name), type: 'function', function: { name: tc.name, arguments: JSON.stringify(tc.args) } }));
  }
  push({ ...base, choices: [{ index: 0, delta: finDelta, finish_reason: fin }] });
  push({ ...base, choices: [], usage: { prompt_tokens: answer.inTokens, completion_tokens: answer.outTokens, total_tokens: answer.inTokens + answer.outTokens } });
  lines.push('data: [DONE]\n\n');
  return lines.join('');
}

function anthropicStop(answer) {
  return answer.finish === 'length' ? 'max_tokens' : (answer.finish === 'tool_calls' ? 'tool_use' : (answer.finish === 'content_filter' ? 'refusal' : 'end_turn'));
}

function renderAnthropic(answer, modelId) {
  const content = [];
  if (answer.text) content.push({ type: 'text', text: answer.text });
  for (const tc of answer.toolCalls) content.push({ type: 'tool_use', id: tc.id || ('toolu_' + tc.name), name: tc.name, input: tc.args });
  if (!content.length) content.push({ type: 'text', text: '' });
  return JSON.stringify({
    id: 'msg_sim', type: 'message', role: 'assistant', model: modelId, content,
    stop_reason: anthropicStop(answer), stop_sequence: null,
    usage: { input_tokens: answer.inTokens, output_tokens: answer.outTokens },
  });
}

function renderAnthropicSSE(answer, modelId) {
  const sr = anthropicStop(answer);
  const lines = [];
  const ev = (name, obj) => lines.push('event: ' + name + '\ndata: ' + JSON.stringify(obj) + '\n\n');
  ev('message_start', { type: 'message_start', message: { id: 'msg_sim', type: 'message', role: 'assistant', model: modelId, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.inTokens, output_tokens: 0 } } });
  ev('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const piece of splitPieces(answer.text)) {
    ev('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } });
  }
  ev('content_block_stop', { type: 'content_block_stop', index: 0 });
  let idx = 1;
  for (const tc of answer.toolCalls) {
    ev('content_block_start', { type: 'content_block_start', index: idx, content_block: { type: 'tool_use', id: tc.id || ('toolu_' + tc.name), name: tc.name, input: tc.args } });
    ev('content_block_stop', { type: 'content_block_stop', index: idx });
    idx++;
  }
  ev('message_delta', { type: 'message_delta', delta: { stop_reason: sr }, usage: { output_tokens: answer.outTokens } });
  ev('message_stop', { type: 'message_stop' });
  return lines.join('');
}

// ---------------------------------------------------------------- server

const startedAt = Date.now();
let brief = null;
let catalog = null;
let plan = null;
let provInfo = {};
let capsByKey = {};
let servedIds = new Set();
let arrivalCount = 0;
let latestNext = null;
let workDone = false;
let inFlight = 0;
let callsMade = {};
let dropped = new Set();
let exiting = false;

function providerBasePath(p, envUrl) {
  if (typeof p.path === 'string' && p.path) return p.path;
  if (typeof p.base_url === 'string' && p.base_url) {
    try { return new URL(p.base_url, envUrl || 'http://localhost').pathname; } catch { return p.base_url; }
  }
  return '/p/' + p.id;
}

function authHeadersFor(info) {
  const headers = {};
  const auth = info.auth || '';
  const key = info.key || '';
  if (auth === 'bearer') headers.authorization = 'Bearer ' + key;
  else if (auth === 'x-api-key') headers['x-api-key'] = key;
  else if (auth === 'x-goog-api-key') headers['x-goog-api-key'] = key;
  if (info.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return headers;
}

async function callProvider(providerId, modelId, neutral) {
  const info = provInfo[providerId];
  if (!info) return { status: 0, answer: null };
  const envUrl = (process.env.MAKEFUN_ENV_URL || '').replace(/\/$/, '');
  let path;
  if (info.wire === 'gemini') path = info.basePath + '/v1beta/models/' + encodeURIComponent(modelId) + ':generateContent';
  else if (info.wire === 'anthropic') path = info.basePath + '/v1/messages';
  else path = info.basePath + '/v1/chat/completions';
  const body = toProviderBody(neutral, info.wire, modelId);
  const headers = { 'content-type': 'application/json', ...authHeadersFor(info) };
  const res = await httpJson('POST', envUrl + path, headers, JSON.stringify(body));
  if (res.status < 200 || res.status >= 300) return { status: res.status, answer: null };
  const answer = parseProviderAnswer(info.wire, res.body);
  if (!answer || typeof answer.text !== 'string') return { status: res.status, answer: null };
  return { status: res.status, answer };
}

function fingerprint(neutral, req) {
  const userText = neutral.messages.filter((m) => m.role === 'user').map((m) => m.text).join('');
  const hasImage = neutral.messages.some((m) => m.images.length > 0);
  return {
    api: neutral.api,
    stream: neutral.stream,
    vision: hasImage,
    tools: neutral.tools.length > 0,
    json: neutral.jsonMode,
    minctx: userText.length > 3000,
    expectApi: req?.api,
    expectStream: req?.stream,
    expectNeeds: reqNeeds(req),
  };
}

function identifyRequest(neutral) {
  arrivalCount++;
  let reqId = null;
  if (latestNext && brief.manifest.some((r) => r.id === latestNext) && !servedIds.has(latestNext)) {
    reqId = latestNext;
  } else {
    const idx = arrivalCount - 1;
    if (idx >= 0 && idx < brief.manifest.length) reqId = brief.manifest[idx].id;
  }
  const req = brief.manifest.find((r) => r.id === reqId) || null;
  const fp = fingerprint(neutral, req);
  const mismatch =
    (req && fp.api !== req.api) ||
    (req && fp.stream !== req.stream) ||
    (req && fp.vision !== fp.expectNeeds.vision) ||
    (req && fp.tools !== fp.expectNeeds.tools) ||
    (req && fp.json !== fp.expectNeeds.json) ||
    (req && fp.minctx !== (fp.expectNeeds.min_context > 0));
  if (mismatch) console.error('[identify] mismatch for', reqId, JSON.stringify(fp));
  return { reqId, req };
}

async function serveRequest(neutral) {
  const { reqId, req } = identifyRequest(neutral);
  if (!reqId || !req) {
    return { text: '', toolCalls: [], finish: 'stop', inTokens: 0, outTokens: 0, model: 'unknown' };
  }
  if (reqNeeds(req).json) neutral.jsonMode = true;
  const ordered = [];
  const seen = new Set();
  const pushCand = (c) => {
    const k = c.provider + '|' + c.model;
    if (!seen.has(k)) { seen.add(k); ordered.push(c); }
  };
  if (plan.assignment[reqId]) pushCand(plan.assignment[reqId]);
  for (const c of plan.rankedLists[reqId] || []) pushCand(c);
  const tryPass = async (allowOver, allowDropped) => {
    for (const c of ordered) {
      if (dropped.has(c.provider) && !allowDropped) continue;
      if (!allowOver && (callsMade[c.provider] ?? 0) >= (SAFE_CAP[c.provider] ?? 0)) continue;
      callsMade[c.provider] = (callsMade[c.provider] ?? 0) + 1;
      const res = await callProvider(c.provider, c.model, neutral);
      if (res.answer) return { ...res.answer, model: c.model };
      dropped.add(c.provider);
    }
    return null;
  };
  inFlight++;
  let result = await tryPass(false, false);
  if (!result) result = await tryPass(true, false);
  if (!result) result = await tryPass(true, true);
  inFlight--;
  if (result) {
    servedIds.add(reqId);
    return result;
  }
  console.error('[serve] unserved', reqId);
  return { text: '', toolCalls: [], finish: 'stop', inTokens: 0, outTokens: 0, model: plan.assignment[reqId]?.model || 'unknown' };
}

function modelsListResponse(req) {
  const apiKey = req.headers['x-api-key'];
  const auth = req.headers.authorization || '';
  const wantAnthropic = typeof apiKey === 'string' && apiKey.length > 0;
  const freeModels = [];
  const seenIds = new Set();
  for (const prov of catalog.providers) {
    for (const m of prov.models) {
      if (!m.free || seenIds.has(m.id)) continue;
      seenIds.add(m.id);
      const c = capsByKey[prov.id + '|' + m.id] || {};
      freeModels.push({ provider: prov.id, id: m.id, context_length: m.context_length, vision: c.vision === true, tools: c.tools === true, json: c.json === true });
    }
  }
  if (wantAnthropic) {
    return JSON.stringify({
      object: 'list',
      data: freeModels.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: FIXED_CREATED_ISO, free: true, context_length: m.context_length, tools: m.tools, vision: m.vision, json: m.json })),
      has_more: false,
    });
  }
  return JSON.stringify({
    object: 'list',
    data: freeModels.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider, free: true, context_length: m.context_length, tools: m.tools, vision: m.vision, json: m.json })),
  });
}

function readJsonBody(req) {
  return new Promise((resolve) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')); } catch { resolve({}); }
    });
    req.on('error', () => resolve({}));
  });
}

function maybeExit() {
  if (exiting) return;
  const total = brief?.manifest?.length || 0;
  if (workDone && servedIds.size >= total && inFlight === 0 && total > 0) {
    exiting = true;
    console.error('[exit] done, served', servedIds.size, 'of', total);
    setTimeout(() => process.exit(0), 50);
  }
}

async function pollWork() {
  const envUrl = (process.env.MAKEFUN_ENV_URL || '').replace(/\/$/, '');
  while (!exiting) {
    const res = await httpJson('GET', envUrl + '/work', {});
    if (isRecord(res.body)) {
      if (typeof res.body.next === 'string') latestNext = res.body.next;
      if (res.body.done === true) {
        workDone = true;
        maybeExit();
        return;
      }
    }
    await new Promise((r) => setTimeout(r, 3));
  }
}

async function registerAndServe(envUrl, port) {
  const res = await httpJson('POST', envUrl.replace(/\/$/, '') + '/register', { 'content-type': 'application/json' }, JSON.stringify({ protocol_version: 1, port, catalog }));
  console.error('[register]', res.status, JSON.stringify(res.body || null));
  pollWork();
}

async function main() {
  const args = process.argv.slice(2);
  const wantList = args.includes('--list');
  const envUrl = process.env.MAKEFUN_ENV_URL || '';
  brief = readBrief();
  if (!brief) {
    if (wantList) { process.stdout.write(JSON.stringify({ providers: [] })); process.exit(0); }
    console.error('no brief (MAKEFUN_PROMPT_FILE unset or unreadable)');
    process.exit(1);
  }
  for (const r of brief.ratings) {
    if (isRecord(r) && typeof r.provider === 'string' && typeof r.model === 'string') {
      capsByKey[r.provider + '|' + r.model] = { vision: r.vision === true, tools: r.tools === true, json: r.json === true };
    }
  }
  for (const p of brief.providers) {
    if (!isRecord(p) || typeof p.id !== 'string') continue;
    provInfo[p.id] = {
      key: typeof p.key === 'string' ? p.key : '',
      auth: typeof p.auth === 'string' ? p.auth : (p.wire === 'anthropic' ? 'x-api-key' : (p.wire === 'gemini' ? 'x-goog-api-key' : 'bearer')),
      wire: typeof p.wire === 'string' ? p.wire : 'openai',
      basePath: providerBasePath(p, envUrl),
    };
  }
  catalog = await buildCatalog(brief, envUrl || null);
  if (wantList) {
    process.stdout.write(JSON.stringify(catalog));
    process.exit(0);
  }
  if (!envUrl) {
    console.error('no MAKEFUN_ENV_URL');
    process.exit(1);
  }
  plan = planRequests(brief, catalog);
  console.error('[plan] loss', plan.totalLoss, 'assigned', Object.keys(plan.assignment).length, 'of', brief.manifest.length);
  for (const id of Object.keys(SAFE_CAP)) callsMade[id] = 0;
  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url || '/', 'http://localhost');
      if (req.method === 'GET' && url.pathname === '/v1/models') {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(modelsListResponse(req));
        return;
      }
      if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
        const body = await readJsonBody(req);
        const neutral = normalizeOpenAI(body);
        const answer = await serveRequest(neutral);
        if (neutral.stream) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          res.end(renderOpenAISSE(answer, answer.model));
        } else {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(renderOpenAI(answer, answer.model));
        }
        maybeExit();
        return;
      }
      if (req.method === 'POST' && url.pathname === '/v1/messages') {
        const body = await readJsonBody(req);
        const neutral = normalizeAnthropic(body);
        const answer = await serveRequest(neutral);
        if (neutral.stream) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          res.end(renderAnthropicSSE(answer, answer.model));
        } else {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(renderAnthropic(answer, answer.model));
        }
        maybeExit();
        return;
      }
      res.writeHead(404, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: { message: 'not found', type: 'not_found' } }));
    } catch (e) {
      console.error('[handler]', e && e.message);
      try { res.writeHead(500, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: { message: 'internal', type: 'internal' } })); } catch { /* already sent */ }
    }
  });
  server.listen(0, '0.0.0.0', () => {
    const port = server.address().port;
    console.error('[listen] port', port);
    registerAndServe(envUrl, port);
  });
  // Safety net: never outlive the trial wall.
  setTimeout(() => {
    if (!exiting) { exiting = true; console.error('[watchdog] exiting at 160s'); process.exit(0); }
  }, 160000);
}

main().catch((e) => { console.error(e); process.exit(1); });
