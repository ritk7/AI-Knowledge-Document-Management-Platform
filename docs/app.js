/* ============================================================
   Retrieval Instrument — browser port of the real pipeline.

   Nothing here is pre-baked. Every number on screen is computed
   in this tab from the same algorithms the Python service runs:

     · RecursiveCharacterTextSplitter (chunk_size 1000 / overlap 150)
     · Okapi BM25 (k1=1.5, b=0.75, epsilon=0.25) — rank_bm25 semantics,
       including the negative-IDF epsilon floor
     · all-MiniLM-L6-v2 sentence embeddings (ONNX, int8-quantised)
     · min-max fusion:  alpha * vec_norm + (1 - alpha) * bm25_norm
     · ms-marco-MiniLM-L-6-v2 cross-encoder re-ranking, sigmoid(logit)
     · two-signal abstention gate: rerank < 0.15 AND cosine < 0.30
   ============================================================ */

const CFG = {
  chunkSize: 1000,
  chunkOverlap: 150,
  candidatePool: 20,
  rerankPool: 10,
  k1: 1.5,
  b: 0.75,
  epsilon: 0.25,
  tRerank: 0.15,
  tVector: 0.30,
  embedModel: "Xenova/all-MiniLM-L6-v2",
  rerankModel: "Xenova/ms-marco-MiniLM-L-6-v2",
};

const DOC_COLORS = ["#C6F24E", "#52D6C8", "#FFB25C", "#EBC66A", "#FF7A6B", "#9D8DF1"];

const state = {
  chunks: [],          // { id, docId, text, chunkIndex, page, vec, tokens }
  docs: [],            // { id, title, text, chunks, color }
  bm25: null,
  pca: null,
  embedder: null,
  tokenizer: null,
  reranker: null,
  ready: false,
  rerankerReady: false,
  last: null,
  evalData: null,
};

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ------------------------------------------------------------------
   1. Tokenizer — mirrors Python's  [a-z0-9]+(?:[-_][a-z0-9]+)*
   Keeping `e-4021` and `api_key` as single tokens is the whole point:
   splitting on the hyphen would turn one rare high-IDF term into two
   common ones and throw the signal away.
------------------------------------------------------------------ */
const TOKEN_RE = /[a-z0-9]+(?:[-_][a-z0-9]+)*/g;
const tokenize = (text) => text.toLowerCase().match(TOKEN_RE) || [];

/* ------------------------------------------------------------------
   2. RecursiveCharacterTextSplitter — faithful port
------------------------------------------------------------------ */
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function splitWithRegex(text, sep, keepSeparator) {
  let splits;
  if (sep) {
    if (keepSeparator) {
      const parts = text.split(new RegExp(`(${escapeRe(sep)})`));
      splits = [parts[0]];
      for (let i = 1; i < parts.length; i += 2) {
        splits.push((parts[i] ?? "") + (parts[i + 1] ?? ""));
      }
    } else {
      splits = text.split(new RegExp(escapeRe(sep)));
    }
  } else {
    splits = Array.from(text);
  }
  return splits.filter((s) => s !== "");
}

function mergeSplits(splits, separator, size, overlap) {
  const sepLen = separator.length;
  const docs = [];
  let current = [];
  let total = 0;
  for (const d of splits) {
    const len = d.length;
    if (total + len + (current.length > 0 ? sepLen : 0) > size) {
      if (current.length > 0) {
        const doc = current.join(separator).trim();
        if (doc) docs.push(doc);
        while (
          total > overlap ||
          (total + len + (current.length > 0 ? sepLen : 0) > size && total > 0)
        ) {
          total -= current[0].length + (current.length > 1 ? sepLen : 0);
          current = current.slice(1);
          if (!current.length) break;
        }
      }
    }
    current.push(d);
    total += len + (current.length > 1 ? sepLen : 0);
  }
  const doc = current.join(separator).trim();
  if (doc) docs.push(doc);
  return docs;
}

function splitText(text, separators, size, overlap) {
  const finalChunks = [];
  let separator = separators[separators.length - 1];
  let newSeparators = [];
  for (let i = 0; i < separators.length; i++) {
    const s = separators[i];
    if (s === "") { separator = s; break; }
    if (new RegExp(escapeRe(s)).test(text)) {
      separator = s;
      newSeparators = separators.slice(i + 1);
      break;
    }
  }
  const splits = splitWithRegex(text, separator, true);
  const mergeSep = ""; // keep_separator=True -> separators already embedded
  let good = [];
  for (const s of splits) {
    if (s.length < size) {
      good.push(s);
    } else {
      if (good.length) { finalChunks.push(...mergeSplits(good, mergeSep, size, overlap)); good = []; }
      if (!newSeparators.length) finalChunks.push(s);
      else finalChunks.push(...splitText(s, newSeparators, size, overlap));
    }
  }
  if (good.length) finalChunks.push(...mergeSplits(good, mergeSep, size, overlap));
  return finalChunks;
}

/* Page mapping: concatenate, record offsets, binary-search a chunk's
   start index back to a page — the same trick the Python side uses so
   chunks stay semantically whole but still carry a page number. */
function chunkDocument(text) {
  const raw = splitText(text, ["\n\n", "\n", ". ", " ", ""], CFG.chunkSize, CFG.chunkOverlap);
  const out = [];
  let index = -1, prevLen = 0;
  for (const c of raw) {
    const t = c.trim();
    if (!t) continue;
    const offset = index + prevLen - CFG.chunkOverlap;
    const found = text.indexOf(c, Math.max(0, offset));
    index = found === -1 ? Math.max(0, offset) : found;
    prevLen = c.length;
    out.push({ text: t, start: index });
  }
  return out;
}

/* ------------------------------------------------------------------
   3. Okapi BM25 — rank_bm25 semantics incl. negative-IDF epsilon floor
------------------------------------------------------------------ */
class BM25 {
  constructor(corpusTokens) {
    this.docFreqs = [];
    this.docLen = [];
    this.idf = new Map();
    const nd = new Map();
    let totalLen = 0;

    for (const tokens of corpusTokens) {
      this.docLen.push(tokens.length);
      totalLen += tokens.length;
      const freq = new Map();
      for (const t of tokens) freq.set(t, (freq.get(t) || 0) + 1);
      this.docFreqs.push(freq);
      for (const t of freq.keys()) nd.set(t, (nd.get(t) || 0) + 1);
    }

    this.N = corpusTokens.length;
    this.avgdl = this.N ? totalLen / this.N : 0;

    let idfSum = 0;
    const negatives = [];
    for (const [word, freq] of nd) {
      const idf = Math.log(this.N - freq + 0.5) - Math.log(freq + 0.5);
      this.idf.set(word, idf);
      idfSum += idf;
      if (idf < 0) negatives.push(word);
    }
    const averageIdf = this.idf.size ? idfSum / this.idf.size : 0;
    const eps = CFG.epsilon * averageIdf;
    for (const w of negatives) this.idf.set(w, eps);
  }

  scores(queryTokens) {
    const out = new Float64Array(this.N);
    for (const q of queryTokens) {
      const idf = this.idf.get(q);
      if (idf === undefined) continue;
      for (let i = 0; i < this.N; i++) {
        const f = this.docFreqs[i].get(q) || 0;
        if (!f) continue;
        const denom = f + CFG.k1 * (1 - CFG.b + (CFG.b * this.docLen[i]) / this.avgdl);
        out[i] += idf * ((f * (CFG.k1 + 1)) / denom);
      }
    }
    return out;
  }
}

/* ------------------------------------------------------------------
   4. Fusion — min-max within the candidate set, then alpha weighting.
   BM25 is unbounded and corpus-dependent; cosine is bounded [0,1].
   Without normalising first, alpha would not mean what it claims.
------------------------------------------------------------------ */
function minMax(map) {
  const vals = [...map.values()];
  if (!vals.length) return new Map();
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const out = new Map();
  if (hi - lo < 1e-9) { for (const k of map.keys()) out.set(k, 1); return out; }
  for (const [k, v] of map) out.set(k, (v - lo) / (hi - lo));
  return out;
}

/* ------------------------------------------------------------------
   5. PCA (power iteration) — real projection of the live embedding
   space, recomputed whenever the corpus changes.
------------------------------------------------------------------ */
function computePCA(vectors) {
  const n = vectors.length, d = vectors[0].length;
  const mean = new Float64Array(d);
  for (const v of vectors) for (let i = 0; i < d; i++) mean[i] += v[i];
  for (let i = 0; i < d; i++) mean[i] /= n;

  const X = vectors.map((v) => {
    const r = new Float64Array(d);
    for (let i = 0; i < d; i++) r[i] = v[i] - mean[i];
    return r;
  });

  const topComponent = (M) => {
    let v = new Float64Array(d);
    for (let i = 0; i < d; i++) v[i] = Math.sin(i * 12.9898) * 43758.5453 % 1;
    let norm = Math.hypot(...v);
    for (let i = 0; i < d; i++) v[i] /= norm;
    for (let iter = 0; iter < 80; iter++) {
      const w = new Float64Array(d);
      for (const row of M) {
        let dot = 0;
        for (let i = 0; i < d; i++) dot += row[i] * v[i];
        for (let i = 0; i < d; i++) w[i] += dot * row[i];
      }
      norm = Math.hypot(...w);
      if (norm < 1e-12) break;
      for (let i = 0; i < d; i++) v[i] = w[i] / norm;
    }
    return v;
  };

  const pc1 = topComponent(X);
  const X2 = X.map((row) => {
    let dot = 0;
    for (let i = 0; i < d; i++) dot += row[i] * pc1[i];
    const r = new Float64Array(d);
    for (let i = 0; i < d; i++) r[i] = row[i] - dot * pc1[i];
    return r;
  });
  const pc2 = topComponent(X2);
  return { mean, pc1, pc2 };
}

/* Andrew's monotone chain — used to draw each document as a region of the
   embedding space rather than a scatter of unrelated dots. */
function convexHull(points) {
  if (points.length < 3) return points;
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop(); upper.pop();
  return lower.concat(upper);
}

/* Expand a hull outward from its centroid so the region reads as a soft
   territory around its points instead of clipping through them. */
function padHull(hull, pad) {
  if (hull.length < 3) return hull;
  const cx = hull.reduce((s, p) => s + p[0], 0) / hull.length;
  const cy = hull.reduce((s, p) => s + p[1], 0) / hull.length;
  return hull.map(([x, y]) => {
    const dx = x - cx, dy = y - cy;
    const d = Math.hypot(dx, dy) || 1;
    return [x + (dx / d) * pad, y + (dy / d) * pad];
  });
}

function project(vec, pca) {
  let a = 0, b = 0;
  for (let i = 0; i < vec.length; i++) {
    const c = vec[i] - pca.mean[i];
    a += c * pca.pc1[i];
    b += c * pca.pc2[i];
  }
  return [a, b];
}

/* ------------------------------------------------------------------
   6. Model loading
------------------------------------------------------------------ */
function setStatus(text, state_, pct) {
  const el = $("status");
  el.dataset.state = state_ || "";
  $("statusText").textContent = text;
  if (pct != null) $("statusFill").style.width = `${clamp(pct, 0, 100)}%`;
}

async function loadModels() {
  try {
    setStatus("fetching runtime…", "", 3);
    const T = await import("https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2");
    T.env.allowLocalModels = false;

    let embedPct = 0;
    setStatus("loading embedder…", "", 8);
    state.embedder = await T.pipeline("feature-extraction", CFG.embedModel, {
      quantized: true,
      progress_callback: (p) => {
        if (p.status === "progress" && p.progress != null) {
          embedPct = p.progress;
          setStatus(`embedder ${Math.round(p.progress)}%`, "", 8 + p.progress * 0.42);
        }
      },
    });

    setStatus("indexing corpus…", "", 52);
    await buildIndex();
    state.ready = true;
    $("run").disabled = false;
    setStatus(`${state.chunks.length} chunks indexed`, "ready", 62);

    // Cross-encoder loads after the index is usable, so the page is
    // interactive while the second model streams in.
    setStatus("loading re-ranker…", "ready", 66);
    state.tokenizer = await T.AutoTokenizer.from_pretrained(CFG.rerankModel);
    state.reranker = await T.AutoModelForSequenceClassification.from_pretrained(CFG.rerankModel, {
      quantized: true,
      progress_callback: (p) => {
        if (p.status === "progress" && p.progress != null) {
          setStatus(`re-ranker ${Math.round(p.progress)}%`, "ready", 66 + p.progress * 0.33);
        }
      },
    });
    state.rerankerReady = true;
    $("rerank").disabled = false;
    setStatus(`ready · ${state.chunks.length} chunks`, "ready", 100);
  } catch (err) {
    console.error(err);
    setStatus("model load failed — BM25 only", "error", 100);
    // Degrade honestly rather than silently: keyword search still works.
    await buildIndex(true);
    state.ready = true;
    $("run").disabled = false;
    $("rerank").checked = false;
    $("rerank").disabled = true;
  }
}

async function embed(texts) {
  const out = [];
  for (const t of texts) {
    const r = await state.embedder(t, { pooling: "mean", normalize: true });
    out.push(Float32Array.from(r.data));
  }
  return out;
}

async function rerankPairs(query, passages) {
  const inputs = state.tokenizer(Array(passages.length).fill(query), {
    text_pair: passages,
    padding: true,
    truncation: true,
  });
  const { logits } = await state.reranker(inputs);
  return Array.from(logits.data).map((x) => 1 / (1 + Math.exp(-x)));
}

/* ------------------------------------------------------------------
   7. Index construction
------------------------------------------------------------------ */
async function buildIndex(skipVectors = false) {
  const raw = await fetch("data/corpus.json").then((r) => r.json());
  state.docs = raw.map((d, i) => ({ ...d, color: DOC_COLORS[i % DOC_COLORS.length], chunks: 0 }));
  state.chunks = [];

  for (const doc of state.docs) {
    const pieces = chunkDocument(doc.text);
    doc.chunks = pieces.length;
    pieces.forEach((p, idx) => {
      state.chunks.push({
        id: `${doc.id}_${idx}`,
        docId: doc.id,
        docTitle: doc.title,
        color: doc.color,
        text: p.text,
        chunkIndex: idx,
        page: 1,
        tokens: tokenize(p.text),
      });
    });
  }

  if (!skipVectors) {
    const vecs = await embed(state.chunks.map((c) => c.text));
    state.chunks.forEach((c, i) => (c.vec = vecs[i]));
    state.pca = computePCA(vecs);
  }
  state.bm25 = new BM25(state.chunks.map((c) => c.tokens));
  renderCorpus();
}

async function addDocument(title, text) {
  const color = DOC_COLORS[state.docs.length % DOC_COLORS.length];
  const doc = { id: `${title}`, title, text, color, chunks: 0 };
  const pieces = chunkDocument(text);
  doc.chunks = pieces.length;
  state.docs.push(doc);

  const added = pieces.map((p, idx) => ({
    id: `${doc.id}_${idx}`,
    docId: doc.id,
    docTitle: title,
    color,
    text: p.text,
    chunkIndex: idx,
    page: 1,
    tokens: tokenize(p.text),
  }));

  if (state.embedder) {
    const vecs = await embed(added.map((c) => c.text));
    added.forEach((c, i) => (c.vec = vecs[i]));
  }
  state.chunks.push(...added);

  // IDF is corpus-wide, so adding a document changes every existing
  // chunk's score — the index has to be rebuilt, not appended to.
  state.bm25 = new BM25(state.chunks.map((c) => c.tokens));
  if (state.embedder) state.pca = computePCA(state.chunks.map((c) => c.vec));
  renderCorpus();
  setStatus(`ready · ${state.chunks.length} chunks`, "ready", 100);
}

/* ------------------------------------------------------------------
   8. The pipeline
------------------------------------------------------------------ */
async function runQuery(q) {
  const alpha = parseFloat($("alpha").value);
  const topK = parseInt($("topk").value, 10);
  const useRerank = $("rerank").checked && state.rerankerReady;
  const timing = {};
  const mark = (k, t) => (timing[k] = performance.now() - t);

  resetStages();

  // -- embed --------------------------------------------------------
  let qvec = null;
  if (state.embedder) {
    busyStage("embed", "···", "encoding");
    const t = performance.now();
    qvec = (await embed([q]))[0];
    mark("embed", t);
    setStage("embed", "384d", `${timing.embed.toFixed(0)} ms`);
  } else {
    setStage("embed", "—", "unavailable");
  }

  // -- lexical ------------------------------------------------------
  const t1 = performance.now();
  const qTokens = tokenize(q);
  const bmRaw = state.bm25.scores(qTokens);
  const bmPairs = [];
  for (let i = 0; i < bmRaw.length; i++) if (bmRaw[i] > 0) bmPairs.push([i, bmRaw[i]]);
  bmPairs.sort((a, b) => b[1] - a[1]);
  const bmTop = bmPairs.slice(0, CFG.candidatePool);
  mark("bm25", t1);
  setStage("lex", String(bmTop.length), `${timing.bm25.toFixed(1)} ms`);

  // -- semantic -----------------------------------------------------
  const t2 = performance.now();
  let vecTop = [];
  if (qvec) {
    const sims = state.chunks.map((c, i) => {
      let dot = 0;
      for (let k = 0; k < qvec.length; k++) dot += qvec[k] * c.vec[k];
      return [i, dot];
    });
    sims.sort((a, b) => b[1] - a[1]);
    vecTop = sims.slice(0, CFG.candidatePool);
  }
  mark("vec", t2);
  setStage("sem", String(vecTop.length), `${timing.vec.toFixed(1)} ms`);

  // -- fuse ---------------------------------------------------------
  const bmMap = new Map(bmTop), vecMap = new Map(vecTop);
  const nBm = minMax(bmMap), nVec = minMax(vecMap);
  const ids = new Set([...bmMap.keys(), ...vecMap.keys()]);
  const fused = [...ids].map((i) => {
    const v = nVec.get(i) ?? 0, b = nBm.get(i) ?? 0;
    return {
      i,
      chunk: state.chunks[i],
      vecRaw: vecMap.get(i) ?? null,
      bmRaw: bmMap.get(i) ?? null,
      vecN: v, bmN: b,
      fused: alpha * v + (1 - alpha) * b,
    };
  });
  fused.sort((a, b) => b.fused - a.fused);
  setStage("fuse", String(fused.length), `α ${alpha.toFixed(2)}`);

  // -- rerank -------------------------------------------------------
  let finalHits, rerankMs = 0;
  const shortlist = fused.slice(0, CFG.rerankPool);
  if (useRerank && shortlist.length) {
    busyStage("rank", "···", `scoring ${shortlist.length} pairs`);
    await new Promise((r) => requestAnimationFrame(r)); // let the frame paint
    const t3 = performance.now();
    const scores = await rerankPairs(q, shortlist.map((h) => h.chunk.text));
    rerankMs = performance.now() - t3;
    shortlist.forEach((h, i) => (h.rerank = scores[i]));
    finalHits = [...shortlist].sort((a, b) => b.rerank - a.rerank).slice(0, topK);
    setStage("rank", finalHits.length ? finalHits[0].rerank.toFixed(3) : "—", `${rerankMs.toFixed(0)} ms`);
  } else {
    finalHits = fused.slice(0, topK);
    setStage("rank", "off", useRerank ? "loading" : "bypassed");
  }

  // -- gate ---------------------------------------------------------
  const rerankConf = finalHits.length && finalHits[0].rerank != null ? finalHits[0].rerank : 0;
  const cosConf = finalHits.length ? Math.max(...finalHits.map((h) => h.vecRaw ?? 0)) : 0;
  const abstain =
    finalHits.length === 0 ||
    (useRerank && rerankConf < CFG.tRerank && cosConf < CFG.tVector);

  const result = {
    q, alpha, topK, useRerank, bmTop, vecTop, fused, finalHits,
    rerankConf, cosConf, abstain, qvec, timing: { ...timing, rerank: rerankMs },
  };
  state.last = result;
  render(result);
  return result;
}

/* ------------------------------------------------------------------
   9. Rendering
------------------------------------------------------------------ */
function resetStages() {
  ["embed", "lex", "sem", "fuse", "rank"].forEach((k) => {
    const el = $(`stage-${k}`);
    el.dataset.on = "0";
    delete el.dataset.busy;
    el.querySelector(".stage__val").textContent = "—";
    el.querySelector(".stage__sub").textContent = "waiting";
  });
}
function setStage(k, val, sub) {
  const el = $(`stage-${k}`);
  el.dataset.on = "1";
  delete el.dataset.busy;
  el.querySelector(".stage__val").textContent = val;
  el.querySelector(".stage__sub").textContent = sub;
}
function busyStage(k, val, sub) {
  const el = $(`stage-${k}`);
  el.dataset.busy = "1";
  el.querySelector(".stage__val").textContent = val;
  el.querySelector(".stage__sub").textContent = sub;
}

function hitCard(h, rank, kind, delta) {
  const score =
    kind === "bm25" ? h.bmRaw?.toFixed(2)
    : kind === "vec" ? h.vecRaw?.toFixed(3)
    : h.rerank != null ? h.rerank.toFixed(3) : h.fused.toFixed(3);
  const c = kind === "bm25" ? "var(--amber)" : kind === "vec" ? "var(--aqua)" : (h.rerank != null ? "var(--gold)" : "var(--lime)");
  let badge = "";
  if (delta === "new") badge = `<span class="hit__delta delta-new">new</span>`;
  else if (delta > 0) badge = `<span class="hit__delta delta-up">▲${delta}</span>`;
  else if (delta < 0) badge = `<span class="hit__delta delta-down">▼${Math.abs(delta)}</span>`;
  return `<div class="hit" style="--c:${c};animation-delay:${rank * 42}ms" data-cid="${h.i}" tabindex="0">
    <div class="hit__top">
      <span class="hit__rank">${rank}</span>
      <span class="hit__src" style="color:${h.chunk.color}">${esc(h.chunk.docTitle)} · ${h.chunk.chunkIndex}</span>
      ${badge}
      <span class="hit__score">${score ?? "—"}</span>
    </div>
    <div class="hit__text">${esc(h.chunk.text.slice(0, 130))}</div>
  </div>`;
}

function render(r) {
  const byId = new Map(r.fused.map((h) => [h.i, h]));

  const bmCards = r.bmTop.slice(0, 6).map(([i], n) => hitCard(byId.get(i), n + 1, "bm25")).join("");
  $("colBm").innerHTML = bmCards || `<div class="empty">no lexical overlap</div>`;
  $("metaBm").textContent = `${r.bmTop.length} hit${r.bmTop.length === 1 ? "" : "s"}`;

  const vecCards = r.vecTop.slice(0, 6).map(([i], n) => hitCard(byId.get(i), n + 1, "vec")).join("");
  $("colVec").innerHTML = vecCards || `<div class="empty">embedder unavailable</div>`;
  $("metaVec").textContent = `${r.vecTop.length} hit${r.vecTop.length === 1 ? "" : "s"}`;

  // Final column shows rank movement relative to pre-rerank fusion order.
  const fusedOrder = new Map(r.fused.map((h, n) => [h.i, n]));
  $("colFinal").innerHTML = r.finalHits
    .map((h, n) => {
      const before = fusedOrder.get(h.i);
      const delta = r.useRerank && before != null ? before - n : 0;
      return hitCard(h, n + 1, "final", delta);
    })
    .join("") || `<div class="empty">nothing retrieved</div>`;
  $("metaFinal").textContent = r.useRerank ? `re-ranked · top ${r.finalHits.length}` : `fused · top ${r.finalHits.length}`;

  renderVerdict(r);
  // Cards animate in with a stagger, and getBoundingClientRect reflects the
  // in-flight transform — so wait for the last one to settle before measuring.
  const settle = 42 * 8 + 440;
  requestAnimationFrame(() => drawConnectors(r));
  clearTimeout(state._wireT);
  state._wireT = setTimeout(() => drawConnectors(r), settle);
  drawMap(r);
  drawGate(r);
  wireHits();
}

function renderVerdict(r) {
  const el = $("verdict");
  el.dataset.mode = r.abstain ? "abstain" : "answer";
  const gate = `<code>rerank ${r.rerankConf.toFixed(3)}</code> ${r.rerankConf < CFG.tRerank ? "&lt;" : "≥"} <code>${CFG.tRerank}</code> · <code>cosine ${r.cosConf.toFixed(3)}</code> ${r.cosConf < CFG.tVector ? "&lt;" : "≥"} <code>${CFG.tVector}</code>`;
  if (r.abstain) {
    el.innerHTML = `<div class="verdict__icon">∅</div><div>
      <div class="verdict__title">Abstains — would not call the model</div>
      <div class="verdict__body">Both confidence signals are below threshold, so the pipeline refuses rather than handing weak context to a language model that would answer fluently anyway. ${gate}</div></div>`;
  } else {
    const why = !r.useRerank
      ? "Re-ranking is off, so the gate falls back to raw cosine similarity."
      : r.rerankConf >= CFG.tRerank
        ? "The cross-encoder is confident in the top passage."
        : "The cross-encoder scores this low — it does that on correct-but-paraphrased passages — but cosine similarity rescues it. That is the whole reason the gate needs both signals.";
    el.innerHTML = `<div class="verdict__icon">→</div><div>
      <div class="verdict__title">Passes the gate — ${r.finalHits.length} chunk${r.finalHits.length === 1 ? "" : "s"} would go to the model</div>
      <div class="verdict__body">${why} ${gate}</div></div>`;
  }
}

/* Bezier connectors: where each final chunk actually came from. */
function drawConnectors(r) {
  const svg = $("wires");
  const box = $("results").getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
  svg.style.height = `${box.height}px`;

  const centreOf = (col, cid) => {
    const el = document.querySelector(`#${col} .hit[data-cid="${cid}"]`);
    if (!el) return null;
    const b = el.getBoundingClientRect();
    return { x: b.right - box.left, y: b.top + b.height / 2 - box.top, left: b.left - box.left };
  };

  let paths = "";
  r.finalHits.forEach((h) => {
    const target = document.querySelector(`#colFinal .hit[data-cid="${h.i}"]`);
    if (!target) return;
    const tb = target.getBoundingClientRect();
    const tx = tb.left - box.left, ty = tb.top + tb.height / 2 - box.top;
    for (const [col, color] of [["colBm", "var(--amber)"], ["colVec", "var(--aqua)"]]) {
      const s = centreOf(col, h.i);
      if (!s) continue;
      const dx = (tx - s.x) * 0.5;
      paths += `<path d="M ${s.x} ${s.y} C ${s.x + dx} ${s.y}, ${tx - dx} ${ty}, ${tx} ${ty}"
        fill="none" stroke="${color}" stroke-width="1.3" opacity=".42" data-cid="${h.i}"
        stroke-dasharray="5 4" class="wire"/>`;
    }
  });
  svg.innerHTML = paths;
}

function wireHits() {
  document.querySelectorAll(".hit").forEach((el) => {
    const cid = el.dataset.cid;
    el.onmouseenter = () => {
      document.querySelectorAll(`.hit[data-cid="${cid}"]`).forEach((n) => n.classList.add("is-linked"));
      document.querySelectorAll(".wire").forEach((w) => {
        w.setAttribute("opacity", w.dataset.cid === cid ? ".95" : ".07");
        if (w.dataset.cid === cid) w.setAttribute("stroke-width", "2");
      });
      document.querySelectorAll(`.map-pt[data-cid="${cid}"]`).forEach((n) => n.setAttribute("r", 7));
    };
    el.onmouseleave = () => {
      document.querySelectorAll(".hit").forEach((n) => n.classList.remove("is-linked"));
      document.querySelectorAll(".wire").forEach((w) => { w.setAttribute("opacity", ".42"); w.setAttribute("stroke-width", "1.3"); });
      document.querySelectorAll(".map-pt").forEach((n) => n.setAttribute("r", n.dataset.r));
    };
    el.onclick = () => openDrawer(parseInt(cid, 10));
  });
}

/* Embedding-space map — real PCA of the live corpus vectors. */
function drawMap(r) {
  const svg = $("map");
  const skel = $("mapSkeleton");
  if (!state.pca || !r.qvec) {
    if (skel) skel.querySelector("span").textContent = "embedder unavailable";
    return;
  }
  if (skel) skel.remove();
  svg.style.display = "block";
  const W = 560, H = 330, pad = 26;
  const pts = state.chunks.map((c, i) => ({ i, p: project(c.vec, state.pca), c }));
  const qp = project(r.qvec, state.pca);
  const xs = pts.map((p) => p.p[0]).concat(qp[0]);
  const ys = pts.map((p) => p.p[1]).concat(qp[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = (v) => pad + ((v - x0) / (x1 - x0 || 1)) * (W - pad * 2);
  const sy = (v) => H - pad - ((v - y0) / (y1 - y0 || 1)) * (H - pad * 2);

  const top = new Set(r.finalHits.map((h) => h.i));
  const cand = new Set(r.fused.map((h) => h.i));
  const qx = sx(qp[0]), qy = sy(qp[1]);

  let s = `<defs><radialGradient id="qg"><stop offset="0%" stop-color="#C6F24E" stop-opacity=".55"/><stop offset="100%" stop-color="#C6F24E" stop-opacity="0"/></radialGradient></defs>`;
  s += `<rect x="0" y="0" width="${W}" height="${H}" fill="none"/>`;
  for (let g = 1; g < 5; g++) {
    s += `<line x1="${(W / 5) * g}" y1="0" x2="${(W / 5) * g}" y2="${H}" stroke="rgba(232,237,230,.05)"/>`;
    s += `<line x1="0" y1="${(H / 5) * g}" x2="${W}" y2="${(H / 5) * g}" stroke="rgba(232,237,230,.05)"/>`;
  }
  // Document territories, drawn under everything else.
  for (const doc of state.docs) {
    const own = pts.filter((p) => p.c.docId === doc.id).map((p) => [sx(p.p[0]), sy(p.p[1])]);
    if (own.length < 2) continue;
    if (own.length === 2) {
      s += `<line x1="${own[0][0]}" y1="${own[0][1]}" x2="${own[1][0]}" y2="${own[1][1]}"
             stroke="${doc.color}" stroke-width="20" stroke-linecap="round" opacity=".07"/>`;
      continue;
    }
    const hull = padHull(convexHull(own), 15);
    s += `<polygon points="${hull.map((p) => p.join(",")).join(" ")}"
           fill="${doc.color}" opacity=".07" stroke="${doc.color}" stroke-opacity=".18"
           stroke-width="1" stroke-linejoin="round"/>`;
  }

  // links from query to what it actually retrieved
  r.finalHits.forEach((h) => {
    const p = pts[h.i];
    s += `<line x1="${qx}" y1="${qy}" x2="${sx(p.p[0])}" y2="${sy(p.p[1])}" stroke="#C6F24E" stroke-width="1" opacity=".38"/>`;
  });
  pts.forEach((p) => {
    const isTop = top.has(p.i), isCand = cand.has(p.i);
    const rad = isTop ? 5.5 : isCand ? 3.4 : 2.4;
    const op = isTop ? 1 : isCand ? 0.66 : 0.22;
    s += `<circle class="map-pt" data-cid="${p.i}" data-r="${rad}" cx="${sx(p.p[0])}" cy="${sy(p.p[1])}" r="${rad}"
            fill="${p.c.color}" opacity="${op}"><title>${esc(p.c.docTitle)} · chunk ${p.c.chunkIndex}</title></circle>`;
  });
  s += `<circle cx="${qx}" cy="${qy}" r="34" fill="url(#qg)"/>`;
  s += `<circle cx="${qx}" cy="${qy}" r="6" fill="#0A100E" stroke="#C6F24E" stroke-width="2"/>`;
  s += `<line x1="${qx - 13}" y1="${qy}" x2="${qx + 13}" y2="${qy}" stroke="#C6F24E" stroke-width="1" opacity=".7"/>`;
  s += `<line x1="${qx}" y1="${qy - 13}" x2="${qx}" y2="${qy + 13}" stroke="#C6F24E" stroke-width="1" opacity=".7"/>`;
  s += `<text x="${clamp(qx + 18, 0, W - 60)}" y="${clamp(qy - 16, 14, H)}" fill="#C6F24E" font-family="JetBrains Mono" font-size="10">query</text>`;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = s;
}

/* Confidence gate — live query plotted against 45 real measured
   eval points. The shaded corner is the AND-region where it abstains. */
function drawGate(r) {
  const svg = $("gate");
  const W = 560, H = 330, pad = 42;
  const sx = (v) => pad + v * (W - pad * 2);
  const sy = (v) => H - pad - v * (H - pad * 2);
  const gx = sx(CFG.tRerank), gy = sy(CFG.tVector);

  let s = "";
  s += `<rect x="${pad}" y="${gy}" width="${gx - pad}" height="${H - pad - gy}" fill="rgba(255,122,107,.12)" stroke="rgba(255,122,107,.4)" stroke-dasharray="3 3"/>`;
  s += `<text x="${pad + 7}" y="${H - pad - 9}" fill="#FF7A6B" font-family="JetBrains Mono" font-size="9.5">ABSTAIN</text>`;

  // axes
  s += `<line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="rgba(232,237,230,.2)"/>`;
  s += `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" stroke="rgba(232,237,230,.2)"/>`;
  s += `<line x1="${gx}" y1="${pad}" x2="${gx}" y2="${H - pad}" stroke="rgba(255,122,107,.3)" stroke-dasharray="2 4"/>`;
  s += `<line x1="${pad}" y1="${gy}" x2="${W - pad}" y2="${gy}" stroke="rgba(255,122,107,.3)" stroke-dasharray="2 4"/>`;
  s += `<text x="${W / 2}" y="${H - 9}" text-anchor="middle" fill="#64736C" font-family="JetBrains Mono" font-size="10">cross-encoder confidence →</text>`;
  s += `<text x="13" y="${H / 2}" text-anchor="middle" fill="#64736C" font-family="JetBrains Mono" font-size="10" transform="rotate(-90 13 ${H / 2})">cosine similarity →</text>`;
  [0, 0.5, 1].forEach((t) => {
    s += `<text x="${sx(t)}" y="${H - pad + 15}" text-anchor="middle" fill="#64736C" font-family="JetBrains Mono" font-size="9">${t}</text>`;
    s += `<text x="${pad - 8}" y="${sy(t) + 3}" text-anchor="end" fill="#64736C" font-family="JetBrains Mono" font-size="9">${t}</text>`;
  });

  if (state.evalData) {
    const col = { answerable: "#C6F24E", unanswerable: "#FF7A6B", holdout: "#FFB25C" };
    state.evalData.points.forEach((p) => {
      s += `<circle cx="${sx(clamp(p.x, 0, 1))}" cy="${sy(clamp(p.y, 0, 1))}" r="3" fill="${col[p.kind]}" opacity=".42"><title>${p.id} (${p.kind})</title></circle>`;
    });
  }

  if (r) {
    const x = sx(clamp(r.rerankConf, 0, 1)), y = sy(clamp(r.cosConf, 0, 1));
    s += `<circle cx="${x}" cy="${y}" r="16" fill="none" stroke="${r.abstain ? "#FF7A6B" : "#C6F24E"}" stroke-width="1" opacity=".5"><animate attributeName="r" values="9;19;9" dur="2.4s" repeatCount="indefinite"/><animate attributeName="opacity" values=".65;0;.65" dur="2.4s" repeatCount="indefinite"/></circle>`;
    s += `<circle cx="${x}" cy="${y}" r="6" fill="${r.abstain ? "#FF7A6B" : "#C6F24E"}" stroke="#0A100E" stroke-width="2"/>`;
    s += `<text x="${clamp(x + 13, 0, W - 80)}" y="${clamp(y - 12, 16, H)}" fill="${r.abstain ? "#FF7A6B" : "#C6F24E"}" font-family="JetBrains Mono" font-size="10">this query</text>`;
  }
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = s;
}

function renderCorpus() {
  const legend = $("mapLegend");
  if (legend) {
    legend.innerHTML = state.docs
      .map((d) => `<span><i style="background:${d.color}"></i>${esc(d.title)}</span>`)
      .join("");
  }
  $("docList").innerHTML = state.docs
    .map((d) => `<div class="doc"><span class="doc__dot" style="background:${d.color}"></span>
      <span class="doc__name">${esc(d.title)}</span>
      <span class="doc__meta">${d.chunks} chunk${d.chunks === 1 ? "" : "s"}</span></div>`)
    .join("");
  $("corpusMeta").textContent = `${state.docs.length} documents · ${state.chunks.length} chunks`;
}

/* ---------- drawer ---------- */
function openDrawer(cid) {
  const h = state.last?.fused.find((x) => x.i === cid);
  const chunk = state.chunks[cid];
  if (!chunk) return;
  $("drawerTitle").textContent = chunk.docTitle;
  $("drawerSub").textContent = `chunk ${chunk.chunkIndex} · ${chunk.text.length} chars · ${chunk.tokens.length} tokens`;

  const rows = [
    ["bm25 raw", h?.bmRaw, "var(--amber)", h?.bmRaw != null ? Math.min(h.bmRaw / 15, 1) : 0, h?.bmRaw?.toFixed(2)],
    ["cosine", h?.vecRaw, "var(--aqua)", h?.vecRaw ?? 0, h?.vecRaw?.toFixed(3)],
    ["bm25 norm", h?.bmN, "var(--amber)", h?.bmN ?? 0, h?.bmN?.toFixed(3)],
    ["vector norm", h?.vecN, "var(--aqua)", h?.vecN ?? 0, h?.vecN?.toFixed(3)],
    ["fused", h?.fused, "var(--lime)", h?.fused ?? 0, h?.fused?.toFixed(3)],
    ["cross-encoder", h?.rerank, "var(--gold)", h?.rerank ?? 0, h?.rerank?.toFixed(3)],
  ];
  $("drawerScores").innerHTML = rows
    .map(([k, v, c, w, txt]) =>
      `<div class="scorerow"><span class="scorerow__k">${k}</span>
       <span class="scorerow__bar"><span class="scorerow__fill" style="background:${c};width:${v == null ? 0 : clamp(w * 100, 0, 100)}%"></span></span>
       <span class="scorerow__v" style="color:${v == null ? "var(--paper-3)" : c}">${v == null ? "—" : txt}</span></div>`)
    .join("");

  const qTokens = new Set(tokenize(state.last?.q || ""));
  const marked = esc(chunk.text).replace(/[A-Za-z0-9][A-Za-z0-9_-]*/g, (w) =>
    qTokens.has(w.toLowerCase()) ? `<mark>${w}</mark>` : w);
  $("drawerText").innerHTML = marked;

  $("drawer").classList.add("is-open");
  $("scrim").classList.add("is-open");
}
function closeDrawer() {
  $("drawer").classList.remove("is-open");
  $("scrim").classList.remove("is-open");
}

/* ------------------------------------------------------------------
   10. Wiring
------------------------------------------------------------------ */
function go() {
  const q = $("q").value.trim();
  if (!q || !state.ready) return;
  $("run").disabled = true;
  runQuery(q).finally(() => ($("run").disabled = false));
}

document.addEventListener("DOMContentLoaded", async () => {
  $("run").onclick = go;
  $("q").addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });

  $("alpha").oninput = (e) => {
    $("alphaVal").textContent = parseFloat(e.target.value).toFixed(2);
    if (state.last) runQuery(state.last.q);       // live re-fusion, no re-embed
  };
  $("topk").oninput = (e) => {
    $("topkVal").textContent = e.target.value;
    if (state.last) runQuery(state.last.q);
  };
  $("rerank").onchange = () => { if (state.last) runQuery(state.last.q); };

  document.querySelectorAll(".chip[data-q]").forEach((c) => {
    c.onclick = () => { $("q").value = c.dataset.q; go(); };
  });

  $("addDoc").onclick = async () => {
    const text = $("docText").value.trim();
    if (text.length < 40) { $("addHint").textContent = "needs at least 40 characters"; return; }
    $("addDoc").disabled = true;
    $("addHint").textContent = "chunking + embedding…";
    const title = ($("docTitle").value.trim() || "Untitled document").slice(0, 60);
    await addDocument(title, text);
    $("docText").value = ""; $("docTitle").value = "";
    $("addHint").textContent = "indexed — it is now searchable";
    $("addDoc").disabled = false;
    if (state.last) runQuery(state.last.q);
  };

  $("drawerClose").onclick = closeDrawer;
  $("scrim").onclick = closeDrawer;
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  window.addEventListener("resize", () => { if (state.last) { drawConnectors(state.last); } });

  state.evalData = await fetch("data/eval.json").then((r) => r.json()).catch(() => null);
  if (state.evalData) {
    const s = state.evalData.summary;
    const cfg = state.evalData.configs["Hybrid + rerank"];
    $("mP1").textContent = cfg[1].precision.toFixed(3);
    $("mMrr").textContent = cfg[1].mrr.toFixed(3);
    $("mGate").textContent = `${s.tuning_caught}/${s.tuning_total}`;
    $("mHold").textContent = `${s.holdout_caught}/${s.holdout_total}`;
  }
  drawGate(null);

  await loadModels();
});
