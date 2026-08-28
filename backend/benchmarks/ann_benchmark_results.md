| Dataset | Method | Build (ms) | Query mean (ms) | Query p95 (ms) | recall@10 | Memory (MB) |
|--------:|--------|-----------:|----------------:|---------------:|----------:|------------:|
| 100 | Brute force (exact) | 0.7 | 0.02 | 0.021 | 1.0 | 0.15 |
| 100 | Custom HNSW (this repo) | 308.3 | 0.256 | 0.28 | 1.0 | 0.26 |
| 100 | ChromaDB (hnswlib, C++) | 923.4 | 0.56 | 0.827 | 1.0 | 71.61 ᴿ |
| 1000 | Brute force (exact) | 2.0 | 0.063 | 0.108 | 1.0 | 1.51 |
| 1000 | Custom HNSW (this repo) | 6982.9 | 0.463 | 0.491 | 1.0 | 2.59 |
| 1000 | ChromaDB (hnswlib, C++) | 907.8 | 0.766 | 0.831 | 1.0 | 88.42 ᴿ |
| 5000 | Brute force (exact) | 3.1 | 0.141 | 0.172 | 1.0 | 7.58 |
| 5000 | Custom HNSW (this repo) | 54135.7 | 0.824 | 0.9 | 1.0 | 12.98 |
| 5000 | ChromaDB (hnswlib, C++) | 1867.3 | 1.032 | 1.119 | 1.0 | 134.92 ᴿ |

Memory is counted exactly (vectors + graph edges) for the Python indexes. ᴿ = RSS delta, used only for ChromaDB, whose allocation happens in C++ behind an opaque API; it includes ~70 MB of fixed client init that does not scale with dataset size.

**`ef_search` sweep — custom HNSW, n=5000**

| ef_search | Query mean (ms) | recall@10 |
|----------:|----------------:|----------:|
| 10 | 0.298 | 0.978 |
| 16 | 0.365 | 0.995 |
| 32 | 0.531 | 0.999 |
| 64 | 0.808 | 1.0 |
| 128 | 1.612 | 1.0 |
| 256 | 3.294 | 1.0 |
