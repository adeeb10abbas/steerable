# V3-C002 pre-release isolation closure

V3-C002 was registered, and its excluded four-cell smoke block completed, but the required two-lane fixed-observation isolation gate failed before behavioral release. The two independent lanes received the same non-language fixture (`7a109dcd32214daba3cf284fc15fd272e678dde3e511e6e1928fdf2bd131f66c`), the same canonical-left prompt bytes, and sampling seed `12000000`. Each returned a finite `[15, 8]` action array and echoed the registered seed.

The outputs were not byte-identical. Lane A's action SHA-256 was `840e7414aa1c261adefe8d17166ad528cb2631bffd25a98c751c4eb3802c12a3`; lane B's was `0c28a6c90bba8120b7ebe8b23a0932b7716d7fb6513939d916435dcd958a6f65`. The maximum absolute elementwise difference was `0.0013794898986816406`, and the mean absolute difference was `0.0002258223103126511`. The frozen compiler therefore rejected the gate with `fixed-observation lane outputs differ`.

Exactly two excluded isolation requests were made, one per lane. No behavioral episode was run, these requests are excluded from all behavioral denominators, and no retry was performed. No passed two-lane isolation gate or behavioral release exists. The 1,364-cell queue was not launched.

This is an infrastructure/pre-release closure, not a semantic result. V3-C002 provides no estimate of canonical-versus-inverse prompt equivalence and authorizes no semantic claim. The retained machine-readable failure report is `gates/isolation_failure_report.json` (SHA-256 `337bd20e62d3ddb608a677b3cfb7d4e2c3c353f11bbb54b4560008096d025c4e`); its target-side raw rehash receipt is `gates/isolation_target_raw_rehash_receipt.json` (SHA-256 `ba6658afcc5d790fb21f1c440f983176f21d9a9545b359de7b526986d770137c`).
