# Encryption / Masking Overhead Experiment (frozen spec)

**Script:** `exp_overhead_crypto.py`  
**Output:** `results/exp_crypto_overhead.json`, `figures/exp_crypto_overhead.pdf`  
**Paper figure style:** `exp8.py` (8 panels, per-subplot title + (a)–(h))

## Schemes (3)

| Label | Implementation | Role |
|-------|----------------|------|
| **VeriFL** | `baselines/secagg_crypto.py` | VeriFL **privacy layer** ≈ Bonawitz SecAgg (Approach A); user–user P2P for all \(N{-}1\) peers |
| **VLMRS** | `baselines/vlmrs_crypto.py` | Flamingo graph neighbors (~\(3\log_2 N\)); user P2P to neighbors only |
| **HVSA** | `baselines/hvsa_crypto.py` | Single TA, 1× PRG mask; **no** user P2P; CS sees **M=5** edge aggregates |

**Excluded from this benchmark:** user/server LHH, signatures, bilinear pairing verification (scope = masking + agreed CS ingress model).

## Crypto platform

- `utils.KA` (X25519), `utils.PRG` (AES-CTR int masks), `utils.AE` (VLMRS neighbor payloads)
- Timing: `median_ms(5×)` wall clock for crypto + analytical P2P delay

## P2P link model (user–user schemes only)

- 10 Mbps uplink, 20 ms RTT **per peer** (sequential), `crypto_common.user_p2p_cost`

## Sweeps

| Experiment | Fixed | Vary |
|------------|-------|------|
| **Exp-A** | \(d = 10{,}000\) | \(N \in \{500,750,\ldots,2000\}\) |
| **Exp-B** | \(N = 1000\) | \(d \in \{10k,30k,50k,70k,100k\}\) |

## User communication (KB)

- **VeriFL / VLMRS:** \(d \times 8\) B masked upload + measured P2P bytes
- **HVSA:** median `pickle.dumps(masked)` size (~\(d \times 8\) B)

## User computation (ms)

- **VeriFL / VLMRS:** median crypto + analytical P2P transmission time
- **HVSA:** median PRG mask only (TA seed setup **outside** timer)

## Cloud server computation (ms)

- **VeriFL / VLMRS:** median time to sum **N** masked `int64` vectors length \(d\)
- **HVSA:** median time to sum **M=5** edge vectors length \(d\)

(Dropout-time reconstruction is **not** included in this sweep; see dropout table experiment separately.)

## Cloud server communication (MB) — analytical ingress

Aligned with `exp8.py` / complexity Table (masking + recovery artifacts):

- **VeriFL:** \(1.08 \cdot N d \cdot 8 + N(N{-}1) \cdot 64\) B + 100 KB fixed  
  → **O(\(Nd + N^2\))**; models Bonawitz-style share recovery at CS.
- **VLMRS:** \(1.04 \cdot N d \cdot 8 + N \cdot (3\log_2 N) \cdot 64 + N \cdot 512\) B + 50 KB fixed  
  → **O(\(Nd + N\log N\))**; CS still receives **N** user updates; graph saves **user** P2P only.
- **HVSA:** \(M \cdot d \cdot 8\) B, \(M=5\) → **O(\(Md\))**, independent of \(N\).

## Reproduce

```bash
python exp_overhead_crypto.py --repeats 5
python exp_overhead_crypto.py --plot-only   # from existing JSON
```

## Not in scope (do not use for this figure)

- `exp8.py` synthetic curves
- `exp5.py` full HVSA pipeline (TA/Edge LHH) — different experiment
- Dropout-dependent CS recovery (use dedicated dropout benchmark later)

## Paper caption checklist

1. VeriFL curve = **SA masking path**, not full VeriFL verification.  
2. VLMRS CS is **O(Nd)** dominant; **not** O(log N) at cloud.  
3. HVSA CS is **O(Md)**; (d) log-scale may pin HVSA to axis bottom.  
4. Sweeps: \(N\) 500–2000, \(d\) 10k–100k (update text if still saying 50–1000 / 1k–10k).
