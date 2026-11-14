# Zero-Failure Floor Router v2 — free-model-aggregator-01

A Node 22 (stdlib only) server that serves the OpenAI Chat Completions API
and the Anthropic Messages API (JSON + SSE) over the ten simulated free-tier
providers in the trial brief, and answers all 20 workload requests with
exactly **20 upstream calls** (zero refusals) on the practice instance.

## Result (local practice harness, kit's own env + scorer, no Docker)

| trial | valid | upstream calls | score | quality | served | wire | failover | catalog |
|-------|-------|----------------|-------|---------|--------|------|----------|---------|
| 0 (dev board) | 20/20 | **20** | **99** | 98 | 100 | 100 | 100 | 100 |
| 1 | 20/20 | 23 | 99 | 98 | 100 | 100 | 100 | 100 |
| 2 | 20/20 | 28 | 98 | 97 | 100 | 100 | 100 | 100 |

Trial 0: all 233 checks passed, 0 failures. Two runs of trial 0 leave
byte-identical `state.json` exports (replay-safe: no wall clock, no
unseeded randomness; every choice is a function of the brief and the
providers' answers so far).

This matches the live leaderboard's top-100 profile exactly: score 99,
quality 98, 20 upstream calls (59 of the top 100 sit at 20 calls, 41 at
21; none above 21). The previous entry (v1) scored the same 99 but with
25 calls (rank 612 of 2647); v2 closes the gap purely on call count.

## Strategy

Computed once at load, from the brief alone (providers, ratings, workload
manifest) plus the kit's public per-provider budget ranges. It never reads
a trial seed or a failure schedule.

1. **Plan up front, scarcest first.** Requests are assigned in order of
   fewest eligible providers (vision, then min_context, then JSON, then
   tools, then plain), each to the candidate minimizing the marginal
   log-survival cost of one more call to that provider, minus a small
   rating term.
2. **First calls are free, extras are concentrated.** Every provider's
   first call always succeeds, so the plan spends the ten first calls
   freely and concentrates the ten extra calls on a few workhorses under
   per-provider caps: zai 1, openrouter 1, cloudflare 1, gemini 3,
   sambanova 2, cerebras 2, ovh 2, groq 5, kilol 3, pollinations 0.
   One provider is forced to refuse its second call, so leaving a few
   providers at one call each is worth more than spreading the extras
   thin. Pollinations is never used: its best free model (1390) costs
   too much quality.
3. **Quality-guarded.** The plan's projected loss stays under 500 of the
   764 a quality of 98 allows (the final plan lands at 748), leaving
   room for failover downgrades. If every candidate is blocked by cap or
   budget, the fallback relaxes the loss budget but keeps the caps.
4. **Drop on first refusal.** A provider that refuses once is dropped
   for the trial (its schedule failed at that call number; an outage or
   a spent budget would fail every later call anyway). The request is
   answered by the next candidate in a failover order that prefers
   models rated >= 1434 first, each group ordered by marginal risk.
   No same-provider retries: v1's retry-same-model rule is what cost it
   the extra 5 calls.

The workload manifest arrives in order (the env's client sends r01..r20
one at a time, no retries), so the n-th request served is manifest[n]
and the plan is keyed by request id.

## Wire handling (kept from v1, one fix)

Full openai <-> anthropic <-> gemini translation with echo preservation
(system text, last user text, tool names in order, max_tokens, stop
sequences in order, image bytes, JSON mode), fixed timestamps/ids, usage
taken from the provider's answer (echo-formula fallback), OpenAI
streaming always carrying `stream_options.include_usage`.

One fix vs v1: `toGemini`'s Anthropic branch dropped user/assistant
messages whose content was a plain string instead of an array of blocks,
which made the provider echo an empty user text (v1 never hit it because
it reserved Gemini for vision requests, which always carry array
content). v2 routes an Anthropic tools request to Gemini, so string
content is now preserved.

## Files

- `main.ts` — the entry (Node 22, stdlib only; run `node main.ts`).
- `README.md` — this file.
- `LICENSE` — MIT.
- `.makefun/owner.json` — entrant wallet.

## Run locally

```sh
# practice trial 0 (kit at ../makefun-free-model-aggregator-01/kit)
MAKEFUN_ENV_DATA_DIR=<kit>/practice/trial-0/input/env \
MAKEFUN_STATE_DIR=/tmp/fma-state \
  node <kit>/program/env/server.mjs &
MAKEFUN_ENV_URL=http://127.0.0.1:<port> \
MAKEFUN_PROMPT_FILE=<kit>/practice/trial-0/input/prompt.txt \
MAKEFUN_TRIAL_SEED=<trial-0 seed> \
  node main.ts
# then score with <kit>/program/scorer/score.mjs
```

Or use the `run-trial.sh` runner in this directory (env + entry +
scorer, no Docker).
