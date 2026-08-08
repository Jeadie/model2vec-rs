# model2vec-rs encode() optimisation hypotheses

Working notes for speeding up `StaticModel::encode` / `pool_ids`. Each entry is an
**independent, testable hypothesis** — validate and land them one at a time so
each delta is attributable.

All line references are against `src/model.rs` at commit `6f51c7a`.

## Background

A model2vec static embedding is just: tokenize → look up one embedding row per
token → mean-pool → L2-normalize. No transformer forward pass, no offsets, no
masks, no attention. The current `encode` path pays for machinery that a static
embedding doesn't need.

The hot path is two stages:

1. **Tokenization** — `tokenizer.encode_batch_fast(...)` (line 372). Parallel
   across a batch via rayon (inside the `tokenizers` crate), but produces full
   `Encoding` objects (offsets, type IDs, attention mask, special tokens,
   alignment) of which only `get_ids()` is used.
2. **Pooling** — `pool_ids` (lines 402–431), called in a **sequential** `for`
   loop (line 374). Single-threaded, scalar accumulate over an `ndarray` row view.

## Measured baseline (Apple M1 Max, potion-base-2M, 64-dim)

Same-machine, same-session runs vs. a purpose-built comparison implementation
(go-potion). Numbers are relative, not absolute (thermal drift between runs), but
the *shape* is stable.

| Scenario | model2vec-rs | reference (minimal tokenizer) |
|---|---|---|
| Single-query throughput | 3.9 MB/s | 41.8 MB/s |
| Batch @ all cores | 19.8 MB/s | 280.7 MB/s |
| Short-query latency p50 | ~10 µs | ~0.6 µs |

Thread-scaling of the batch path (batch=all, potion-base-2M):

| Threads | MB/s | speedup |
|---|---|---|
| 1 | 4.1 | 1.0× |
| 2 | 7.4 | 1.8× |
| 4 | 12.2 | 3.0× |
| all (10) | 19.8 | 4.8× |

The sublinear thread-scaling (4.8× on 10 cores) is the signature of **H10**:
only tokenization parallelizes; pooling is serial. A fully-parallel pipeline
scales ~linearly to 4 cores on the same box.

---

## Running each hypothesis independently

**H1 and H4 have landed as the only implementation** (no feature flag —
`src/model.rs`'s `tokenize_batch_ids` and `accumulate_rows` are unconditional).
H3 was evaluated and dropped (marginal, not worth the permanent branching —
see Findings below). The mechanism below still applies to whichever of
H2/H5–H10 gets picked up next.

Each remaining hypothesis is validated as a compile-time **feature flag**
(`opt-hN`) that swaps the corresponding code path in the **real**
`StaticModel::encode` — not a copy. Default build (no feature) stays the
current (H1+H4) implementation. Flags are independent and may be combined;
enable one at a time to attribute a delta to a single hypothesis. Once a
hypothesis is validated, land it as the only implementation and retire its
flag (as done for H1/H4) rather than carrying it as permanent optionality.

Measure with the **standard `encode` benchmark** (`benches/encode.rs`), which
runs the real `encode` path (single-query + batch sweep). Verify correctness
with the **existing Python-golden parity test** — no bespoke parity harness,
the real reference:

```bash
# baseline vs flag on — same benchmark, real code path, flag off vs on
# (run the Go benchmark once first to cache big.txt)
cargo bench --bench encode -- 4                     # current default (H1+H4)
cargo bench --bench encode --features opt-hN -- 4   # with a new hypothesis
cargo bench --bench encode --features opt-hN -- 4 minishlab/potion-retrieval-32M

# correctness of the flag, on the real path, against the Python golden:
cargo test --features opt-hN                         # incl. the vocab-quantized path
```

**Adding a hypothesis:** add an `opt-hN` feature in `Cargo.toml`, and a
`#[cfg(feature = "opt-hN")]` alternative of the relevant function in
`src/model.rs`. Only pooling-stage hypotheses (H5, H7, H9) fit the current
`accumulate_rows` seam; encode-stage ones (H2, H6, H8, H10) will each gate
their own call site.

### Results so far — end-to-end `encode` MB/s, best-of-N, Apple M1 Max

Real `encode` throughput, flag off vs on. Parity = the Python-golden test
(`test_rust_matches_python`, standard **and** vocab-quantized) passing with the
flag enabled. Absolutes drift ~±10% run-to-run (thermal); read the ratios.

**potion-base-2M (64-dim):**

| variant | single-query | batch=all |
|---|---|---|
| baseline | 3.3 | 18.0 |
| opt-h4 | 3.5 | 22.6 |
| **opt-h1** | **31.1 (9.3×)** | 31.2 |
| **opt-h1+h4** | **52.3 (15.7×)** | **54.5 (3.0×)** |

**retrieval-32M (512-dim):**

| variant | single-query | batch=all |
|---|---|---|
| baseline | 2.6 | 6.8 |
| opt-h4 | 3.25 | 17.8 (2.6×) |
| opt-h1 | 8.0 (3.1×) | 8.1 |
| **opt-h1+h4** | **31.2 (12.2×)** | **32.4 (4.8×)** |

Parity: **all golden + token-ID tests pass** with `opt-h1`, `opt-h4`, and both
combined (incl. vocab-quantized). H1's token-ID parity test (`fast_wordpiece`)
matches the `tokenizers` crate across ASCII/accents/CJK/punctuation.

**Findings:**
- **H1 (custom IDs-only WordPiece tokenizer) is the biggest single lever**, as
  predicted: single-query 9.3× on 64-dim, 3.1× on 512-dim — largest where
  tokenization dominates (low-dim). It drops `encode_batch_fast`'s
  offset/alignment/mask machinery for the BERT pipeline; non-WordPiece models
  (multilingual/Unigram) fall back to the crate untouched.
- **H4 (contiguous-slice pooling)** is the complementary lever: largest where
  pooling dominates (512-dim batch, 2.6×). H1 and H4 **stack** — combined they
  reach 15.7× (64-dim) and 12.2× (512-dim) single-query.
- **With H1, batch ≈ single-query** (e.g. 31.1 vs 31.2). The custom tokenizer is
  serial (a `.map` over texts), so batching no longer wins the way it did when
  `encode_batch_fast` parallelized tokenization across cores — yet serial H1
  still beats the old *parallel* batch. Next lever: **parallelize the H1 tokenize
  loop + H10 (parallel pooling)** to make batch scale across cores again.
- H3 (~1.02× isolated) is marginal; keep only for quantized models where the
  branch is actually taken.

**Status: H1 and H4 landed as the default (no feature flag); H3 evaluated and
dropped.** The combo validated above (H1+H4) is what `src/model.rs` now always
runs.

**Scope decision — H1 is WordPiece-only; the Unigram/multilingual pipeline stays
on the crate fallback (deliberate).** WordPiece's win came from skipping
`encode_batch_fast`'s Encoding machinery (offsets/alignment) while the actual
tokenization was cheap. The multilingual model is the opposite: its cost is
*inherent* work we can't skip — a 316 KB Precompiled charsmap normalizer and a
Viterbi over a 500 K-entry Unigram vocab. The crate already does both with an
optimized trie (multilingual runs at ~1.45 MB/s; go-potion's hash-based Unigram
was 18× *slower*, showing how easy it is to regress). A custom IDs-only Unigram
would only shave the small Encoding-overhead fraction (est. ~1.4 → ~2 MB/s) for a
multi-day charsmap reimplementation on a single model — not worth it. If ever
revisited, delegate the charsmap to the `spm_precompiled` crate rather than
reimplementing it. `from_tokenizer` returns `None` for any non-WordPiece
pipeline, so these models fall back to the crate unchanged.

**Next levers (real payoff):** with H1 the custom tokenizer is serial, so batch
no longer scales across cores — parallelize the H1 tokenize loop (rayon) + H10
(parallel pooling) to make WordPiece batch throughput scale again.

---

## Hypotheses

### H1 — Emit token IDs only; drop the general-purpose tokenizer from the hot path
- **Status: LANDED as the default** (`src/fast_wordpiece.rs`, WordPiece
  pipeline only; non-WordPiece models fall back to the crate). No feature flag —
  `tokenize_batch_ids` always tries `fast_tok` first. Verified: token-ID
  parity vs the crate + Python-golden tests pass. Result: single-query 9.3×
  (64-dim) / 3.1× (512-dim). See the results table above.
- **Location:** line 372, `encode_batch_fast::<String>(...)`
- **Observation:** `encode_batch_fast` builds full `Encoding` objects (offsets,
  alignment tracking, attention masks, type IDs, special tokens) and we keep
  only `get_ids()`. For static embeddings none of the rest is ever read.
- **Hypothesis:** the discarded work is the single largest per-core cost. A
  purpose-built forward tokenizer (WordPiece / Unigram) that returns only IDs —
  no offsets, no alignment — would recover most of the per-core gap.
- **Change:** replace the `tokenizers`-crate call on the hot path with a minimal
  IDs-only tokenizer (WordPiece greedy-longest-match; Unigram via a trie +
  Viterbi). Keep `tokenizers` for loading `tokenizer.json` config.
- **Impact:** highest. Dominant cost, especially for low-dim models where
  pooling is cheap.
- **Effort/risk:** high — largest change; must match reference token IDs exactly
  across ASCII + non-ASCII. Gate behind a token-ID parity test.

### H2 — Stop cloning the batch into owned `String`s each call
- **Location:** line 372, `truncated.into_iter().map(Into::into).collect()`
- **Observation:** `truncated: Vec<&str>` is immediately re-collected into
  `Vec<String>` (owned) purely to satisfy `encode_batch_fast`'s signature. That
  allocates and copies the entire batch text on every call.
- **Hypothesis:** an avoidable per-batch allocation of O(batch bytes).
- **Change:** use an `encode_batch_fast` overload / input type that accepts
  `&str` (e.g. `EncodeInput`/`&str` inputs), or feed borrowed slices, to skip the
  owned-String materialization.
- **Impact:** low–moderate; grows with batch size.
- **Effort/risk:** low; localized. Verify the tokenizers API accepts borrowed input.

### H3 — Hoist the `weights` / `token_mapping` branches out of the per-token loop
- **Status: EVALUATED, NOT LANDED.** Isolated impact was marginal (~1.02×) and
  the H1+H4 results table above never included H3 — the validated combo is
  H1+H4 alone. Not worth carrying the extra `(weights, token_mapping)` match
  arms permanently for that little; revisit only if a future quantized-model
  workload shows the per-token `Option` lookups actually dominating.
- **Location:** lines 408–414 (inside `pool_ids`)
- **Observation:** every token does two `Option::as_ref().and_then(|s| s.get(tok))`
  — bounds-checked Option lookups — for `token_mapping` and `weights`, *even when
  both are `None`* (the common non-quantized model, e.g. potion-base-2M).
- **Hypothesis:** per-token `Option` + bounds-checked `get` defeats a tight
  accumulate loop and is paid ~1.45M times/run for nothing on plain models.
- **Change:** branch once on `(weights.is_some(), token_mapping.is_some())` and
  run a specialized inner loop per case; the common `(None, None)` case becomes a
  bare `row_idx = tok; sum += row`.
- **Impact:** moderate (bigger on low-dim, where per-token overhead dominates).
- **Effort/risk:** low; output-identical.

### H4 — Accumulate over a contiguous `&[f32]`, not an `ndarray` row view
- **Status: LANDED as the default** (`accumulate_rows` in `src/model.rs`). No
  feature flag. See the results table above.
- **Location:** line 415 `self.embeddings.row(row_idx)`, loop at line 416
- **Observation:** `embeddings` is built with `Array2::from_shape_vec((rows, cols), …)`
  (line 265) → row-major / standard layout, so each row **is** contiguous. But
  `.row()` yields a strided `ArrayView1` whose `.iter()` tends to block
  autovectorization.
- **Hypothesis:** iterating the flat backing slice (`&embeddings_flat[row*dim..][..dim]`)
  lets LLVM autovectorize the `sum += v * scale` loop.
- **Change:** keep a flat `&[f32]` handle to the embeddings (or use
  `.as_slice()` / `.row(i).as_slice()`), index directly, drop the ndarray view on
  the hot path.
- **Impact:** moderate–high, scales with dim (large on 512-dim retrieval-32M).
- **Effort/risk:** low–moderate; pairs naturally with H3 and H5.

### H5 — Explicitly SIMD the accumulate
- **Location:** line 416
- **Observation:** the `sum += v * scale` (or `sum += v` for unweighted) loop is
  the hottest instruction stream and is scalar.
- **Hypothesis:** an explicit SIMD accumulate (portable `std::simd`, or a crate
  like `wide`) beats scalar and is more reliable than hoping for autovectorization.
- **Change:** vectorize the fused-multiply-add over the row; fall back to scalar
  for `dim` below a measured crossover.
- **Impact:** high on high-dim models; smaller on 64-dim where per-call overhead
  dominates.
- **Effort/risk:** moderate; requires a scalar fallback + parity test. Depends on
  H4 (needs contiguous input).

### H6 — Fuse UNK filtering into the accumulate instead of `retain`
- **Location:** line 377 `token_ids.retain(|&id| id as usize != unk_id)`
- **Observation:** UNK removal is a separate compaction pass over the `Vec`
  before pooling.
- **Hypothesis:** the extra pass (and the `Vec` rewrite) is avoidable.
- **Change:** skip `id == unk_id` inline inside the `pool_ids` accumulate loop;
  drop the `retain`. One pass, no compaction.
- **Impact:** low.
- **Effort/risk:** low; output-identical (verify `cnt` still excludes UNK).

### H7 — Fuse the average and normalize passes
- **Location:** lines 421–429
- **Observation:** two full passes over `sum`: divide by `cnt`, then divide by
  `norm`.
- **Hypothesis:** collapsible into one scale by `1/(cnt·norm)`.
- **Change:** compute `norm` of the summed (un-averaged) vector, then a single
  pass multiplying by `1.0 / (denom * norm)` (average scaling cancels in the
  normalized case; keep the two-pass form when `normalize == false`).
- **Impact:** low.
- **Effort/risk:** low; watch float ordering vs. the parity tolerance.

### H8 — Don't `to_vec()` the token IDs per sentence
- **Location:** line 375 `let mut token_ids = encoding.get_ids().to_vec();`
- **Observation:** a heap `Vec<u32>` is allocated per sentence just to filter/
  truncate before pooling.
- **Hypothesis:** avoidable per-sentence allocation (~one per input).
- **Change:** iterate `encoding.get_ids()` directly in `pool_ids`, applying
  UNK-skip (H6) and truncation inline; pass a slice, not an owned `Vec`.
- **Impact:** low–moderate; most visible on short-input / high-QPS latency.
- **Effort/risk:** low; interacts with H6 and the `pool_ids` signature.

### H9 — Reuse the accumulator buffer across sentences
- **Location:** line 404 `let mut sum = vec![0.0_f32; dim];`
- **Observation:** a fresh `dim`-length `Vec<f32>` is allocated for every input.
- **Hypothesis:** avoidable per-sentence allocation; matters for latency and for
  the parallel path (use a thread-local / per-worker scratch buffer).
- **Change:** accumulate into a reusable buffer (zeroed per input); under rayon
  (H10) use a per-thread buffer. The result `Vec` can still be produced once.
- **Impact:** low–moderate (latency-sensitive workloads).
- **Effort/risk:** low; must reset between inputs.

### H10 — Parallelize pooling (make the whole pipeline multi-core)
- **Location:** line 374, the `for encoding in encodings` loop
- **Observation:** tokenization (line 372) is already rayon-parallel, but
  `pool_ids` runs sequentially on one core. This is exactly the sublinear
  thread-scaling measured above (4.8× on 10 cores vs. ~linear for a fully-parallel
  pipeline).
- **Hypothesis:** pooling is a meaningful, fully-parallelizable fraction of batch
  time — larger for high-dim models — left entirely on one core.
- **Change:** `encodings.into_par_iter().map(|e| pool_ids(...)).collect()` (add
  `rayon` as a **direct** dependency — currently only transitive via `tokenizers`).
  Combine with per-thread scratch buffers (H9).
- **Impact:** high for batch throughput, especially 512-dim; recovers the
  batch-scaling gap. No effect on single-query latency.
- **Effort/risk:** low code, but adds a direct `rayon` dep; ensure deterministic
  output ordering (`par_iter` preserves index order on `collect`).

---

## Dependencies between hypotheses

- **H4 → H5:** SIMD needs a contiguous slice first. H4 has landed.
- **H6 + H8:** both change how IDs flow into `pool_ids`; do together.
- **H9 → H10:** parallel pooling wants per-thread scratch buffers.
- With H1 landed, the tokenizer is now serial per text — **H10 should
  parallelize the `FastWordPiece` tokenize loop itself** (not just pooling) to
  make batch throughput scale across cores again (see H1's Findings note above).

## Suggested order (remaining: H2, H5–H10; H1/H4 landed, H3 dropped)

1. **H6, H8, H9** — self-contained, low-risk, output-identical pooling
   cleanups. Measure each with `-benchmem`-style allocation counts + throughput.
2. **H5** — SIMD accumulate on top of the landed H4.
3. **H10** — parallelize both `FastWordPiece::encode_ids` and pooling (biggest
   remaining batch-throughput win, now that H1 made tokenization serial).
4. **H2** — kill the per-batch owned-String clone (only matters for the
   non-WordPiece crate-fallback path now that H1 handles WordPiece directly).

## Validation (must precede any change)

- **Numerical parity:** a golden test asserting encoded vectors match the current
  `main` build within a tight tolerance (max-abs-diff / cosine ≈ 1.0) on a diverse
  corpus (ASCII + non-ASCII, short + long, incl. a quantized model exercising
  `weights`/`token_mapping`). No speed change is worth a silent output drift.
- **Token-ID parity** (gate for **H1**): identical ID streams vs. the reference
  tokenizer, per input, on the same diverse corpus.
- **Attribution:** measure each hypothesis in isolation (single-query throughput,
  batch sweep 1/32/256/1024/all, thread-scaling 1/2/4/all, short-query p50/p90/p99)
  so each delta is traceable to one change.
