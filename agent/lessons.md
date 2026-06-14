
## 2026-06-14 — Autonomous codegen cannot reproduce a validated construction; ablations need hand-edits
Tried to test "does amihud_illiq_tranched_v3's premium survive restricting shorts to borrowable
names" by enqueuing a spec that said "reproduce v3 byte-for-byte + a borrow floor." Two independent
smith runs of the SAME spec gave search Sharpe 0.62 (run#1) and 0.02 (v2); the REAL deployed v3 is
1.89. Three different strategies. The codegen re-derives a fresh, much-weaker Amihud-flavoured book
each time (drift in size-bucketing / tranching / sizing), so the effect under test (the borrow floor)
is swamped by reproduction variance. I briefly mis-reported run#1 as "+150% survival" — it was a
ratio between two internally-broken baselines, both far below the real v3.
RULE: an ABLATION/VARIANT of a VALIDATED strategy must be tested by DETERMINISTICALLY hand-editing
the actual validated module (change exactly the one thing, diff the return stream), NEVER by
re-generating from a prose "reproduce X + change Y" spec. Codegen is for NEW hypotheses; controlled
modifications of existing validated code are a hand-edit + harness re-run. Same family as the
hedging-pressure 3-crash close: autonomous codegen is unreliable for complex/precise reproduction.
