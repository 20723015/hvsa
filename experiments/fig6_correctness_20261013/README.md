# Fig. 6-style secure aggregation correctness run

Run from `E:\hvsa` with the repository's training interpreter:

```powershell
D:\Anaconda\python.exe -u experiments\fig6_correctness_20261013\run_fig6_correctness.py
D:\Anaconda\python.exe experiments\fig6_correctness_20261013\prepare_fig6_outputs.py
```

The first command ran on 2026-10-13 with Python 3.9.15, NumPy 1.26.4, and PyTorch 2.7.0+cu118, configured to train on CPU. Its measured wall time was 68.528 seconds. The run used one seed (20261013), MNIST, Tiny-MNIST (7,850 parameters), 12 clients across 3 edge servers, 24 fixed IID examples per client, one local epoch per communication round, no dropout, and 80 rounds.

For a matching rerun, use the recorded versions above; the experiment also used `ecdsa 0.19.1`. The repository's upload quantization is `(float32_parameters * 1_000_000).astype(int64)`. NumPy 1.26.4 promotes that multiplication to float64, while NumPy 2.3.5 follows different scalar-promotion rules. An independent check of the saved local parameters, converted to float64 before multiplying by 1,000,000, reproduced the saved int64 values at every coordinate. The runner now checks this full-vector equivalence for every client and round.

Only rounds 35 and 79 ran the repository's complete secure upload, LHH aggregation, Ed25519 signing, TA mask reconstruction, and `EdgeServer.verify_global_result` path. The other 78 rounds used ordinary quantized FedAvg. At the sampled rounds, four ES-0 local `fc.weight` arrays are shown for comparison; each secure aggregate covers all 12 clients. The attack probes call the same production verifier once per alteration and sampled round.

This is a reduced-scale experiment using the repository's protocol. It is not a reproduction of the paper's 1,000-client, five-run evaluation or its pairing-based proof construction. See `results.json` for the exact setup and limitations, `results.csv` for per-round and attack outcomes, `protocol.log` for the execution trace, and `summary_audit.txt` for the corrected aggregate count note.
