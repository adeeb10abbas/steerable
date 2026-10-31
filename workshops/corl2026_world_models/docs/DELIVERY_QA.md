# Rewritten paper: delivery verification

Date: 2026-10-31. Status: **working research paper; not submitted**.

## Manuscript and PDF

The paper is now titled **Spatial Instruction Following in World–Action Models**.
It reports the completed Nano and DreamZero position-reflection experiments.
The previous forecast-scoring draft is preserved in commit `8657fa7`; its
supporting analysis remains available but is not presented as the current paper.

`paper/main.pdf` has **four pages total including references**, compiled with the
unchanged official CoRL 2026 style in preprint mode and an anonymous author line.

- PDF SHA256: `58a657b3ab86e36e6dfe6aa979efc9011c9c66700b6717385e66bc773998e965`.
- LaTeX source SHA256: `3abc21d3745781e2896e2834c6f7b62d464f82983f5fe8b3633767c620f0602d`.
- Official style SHA256: `62f38cd8df7ad718796617595c8230da136fc83ea1d7e1ba80084113409ec957`.
- Numerical figure input SHA256: `f437fec607ea969a4c0293b157a74175c410b5f88709ac73262b3035314f1230`.
- Design figure PDF/PNG SHA256: `9896dafc73897ca19fe6558d9385a8b1512cae858ba815c88743642185ea8201` / `9f1dbac8c32485ebdbaa0df80f8dc1b40c36c9369fd118c700c4e591e4cd94df`.
- Results figure PDF/PNG SHA256: `352fc07d916fe0eb69811189ce5a33d3d4b6250ff2b5406616588d92efffcbd2` / `6f1eacb1fccc77a7b70e7b1d393816d9e87c5b570c29384e38d322ebd7ddde98`.
- Automated PDF checks passed: four pages, expected result text, resolved
  references, no overfull boxes or LaTeX errors.
- All four pages were rendered to temporary PNGs and visually inspected. The
  review renders are intentionally not checked in or cited as persistent files.
  Body text, equations, table, figures and references are readable, with no
  clipping, overlap or missing glyphs. The retained analyses, limitations and
  references fit on page 4 without a blank or spill page.

The result figure is generated from `results/paper_results.json`. The experimental
design figure is explicitly labeled as a schematic, not a measured trajectory.

## Numerical verification and independent review

The exact documented disposable environment uses Python 3.10.21, inherits
Torch 2.7.0+cu128 from the qualified any6d environment, and resolves NumPy
2.2.6, matplotlib 3.10.9, pypdf 6.18.1 and jsonschema 4.26.0. The clean
isolated pre-build discovery passed **1030/1030 tests in 129.990 s**
(**132.87 s** wall) for the preceding H13-only package. Its corresponding
protected-tree discovery passed **1037/1037 tests** in 126.000 s (128.43 s
wall; peak resident set size 905,792 KiB); seven of those methods belong to the protected untracked
N3 test and are not package files. After integrating H03/H04,
the next clean isolated discovery passed **1041/1041 tests in 129.042 s**
(**131.87 s** wall; peak resident set size **909,184 KiB**). After integrating
H02, a clean isolated pre-build discovery passed **1048/1048 tests in
128.217 s** (**130.96 s** wall; peak resident set size **913,632 KiB**).
The final Turn2909 authoritative isolated build passed all **19 stages**, including
**1060/1060 tests in 130.452 s**, in **146.35 s** wall time with a peak resident
set size of **915,176 KiB**. The stages cover tests, retained-data analyses,
media packaging and four full MP4 decodes plus verification, paper results, the
retained H01--H13 historical-evidence coverage audit, historical replay, the
private annotation draw, figures and the official-template PDF. A preceding
clean attempt stopped fail-closed when the new audit treated the freshly
regenerated `paper_results.json` checkout as immutable; the correction still
authenticates and parses the pinned base Git object while allowing only that
builder-generated checkout copy to evolve, and adds a regression test for the
boundary. These software qualifications validate the package; they are not
behavioral episodes, model-policy competence or scientific evidence.

The final Turn2910 authoritative isolated build at commit
`5d9c5f7f0611fc3add97d0417221d52214e42c2a` passed all **20 stages**, including
**1070/1070 tests in 133.302 s**, in **150.21 s** wall time with a peak resident
set size of **920,312 KiB**. It produced the four-page, 162,571-byte PDF and
12,981-byte LaTeX source whose hashes are recorded above. Its 563,702-byte/
12,257-line validation log and 186,768-byte/5,480-line build report have SHA256
`6e36d8b8d302044442f091b6bf3623b82c7f15347a45614873b8b5d45b596218` and
`580434f7f4dfeb3afa3f7d797ba97bd80495c52b9916b5bbbb64cd01b216405e`.
The complete 187,516-byte/5,500-line stdout and 801-byte/23-line resource
receipt have SHA256
`d105a77f1fb1cd2ac549bad82a1402ba4209321daace76b5978bae51737eed2f` and
`7af3b9ddf370d563584ae401572fd2cada551036ca9fa38fbe33d28c60f54a4d`.
A preceding isolated attempt passed the same 1,070 tests and 20 stages and made
the same clean four-page PDF, then stopped fail-closed because pypdf removes
spaces around the TeX math phrase `106 $+32$ targets`. The checker correction
accepts only optional whitespace around that exact phrase; focused tests and a
direct check against the failed PDF passed before the clean rerun. Root and an
independent reviewer rendered and inspected every page and both returned GO.
The resulting protected-tree discovery passed **1077/1077 tests in 128.264 s**
(**130.71 s** wall; peak resident set size **922,052 KiB**), including exactly
seven protected-only N3 methods. Its 143,646-byte/1,091-line stdout and
839-byte/23-line resource receipt have SHA256
`a98ac3422d1df61e07c6b4942af11d79512856eb375266811ae89333cb8579d7` and
`ea68a67afeb96ca19bb2624df91e6eb6ae583a256d914fbc7554cb84c1231760`.

The Turn2911 matched-window extension at commit
`1a1602f9dea4715234bb8f5c6495290dcf5f634d` was then rebuilt from a fresh
detached worktree in the same qualified environment. The authoritative build
passed all **20 stages**, including **1071/1071 tests in 137.794 s**, in
**157.42 s** wall time with a peak resident set size of **918,488 KiB**. Its
563,791-byte/12,258-line validation log, 265,033-byte/7,516-line build report,
265,781-byte/7,536-line stdout and 832-byte/23-line resource receipt have
SHA256 `7c720bdd0da23740e8e7aa37c020ac9d4c4a016b0b96766153b57bda74d439a0`,
`bfe3aeda0f3cd3c0436fbe2cbd555666e8265634e9ec9b311fce3a15b43b3bab`,
`2846b5ad42f20fd8305d155ef56146697865f4fba29a539b63af3e7442537e15`
and `53c66509e7c1ed6c4689ba44cd5f71b8c45c4c76db0afc6b9821456d1c3c1a8b`.
A clean preceding attempt passed the first 19 stages and stopped before the
PDF because the pinned TeX wrapper was absent from `PATH`; no result from that
attempt was promoted. The qualified protected-tree discovery subsequently
passed **1078/1078 tests in 137.835 s** (**141.55 s** wall; peak resident set
size **919,012 KiB**), including exactly seven protected-only N3 methods. Its
143,779-byte/1,092-line stdout and 841-byte/23-line resource receipt have
SHA256 `65771a99dface27f724364628ce9a0946efcb27076368dc07d994b65cf396a02`
and `930f5d2ac6cd11450a72478841f559f86f902100df1a170dfde1dd9e1de03bbb`.
The four-page PDF and LaTeX source are byte-identical to the Turn2910
artifacts, and all four freshly rendered pages passed visual review. Two
independent reviewers and the root review returned GO within the closed claim
boundary.

The Turn2912 fixed-request-index extension at commit
`524b6d876651dbedb211367502a4d77efdd1d578` was rebuilt from a new detached
worktree in the same qualified environment. The authoritative build passed all
**20 stages**, including **1071/1071 tests in 137.601 s**, in **157.21 s** wall
time with a peak resident set size of **924,392 KiB**. Its 563,725-byte/12,258-line
validation log, 329,693-byte/9,772-line build report, 330,441-byte/9,792-line
stdout and 834-byte/23-line resource receipt have SHA256
`70b0af3e038cfed319f1a4a5983e2b5e6d2758f212f6c972edae569425042396`,
`bc1ed95e16b13c87b10b40d0bcbee955a1921df5c60397f640decb1f87c13e7a`,
`a5f43ac10a82a894ad0eafddab6e26b83b7778968cdf59f460a954d16976dfba`
and `06481e4ecf15063a1f4a891893b593dd29a146b76fc52a03b224843bc482ea3b`.
The resulting protected-tree discovery passed **1078/1078 tests in 137.496 s**
(**141.21 s** wall; peak resident set size **921,476 KiB**), including exactly
seven protected-only N3 methods. Its 143,779-byte/1,092-line stdout and
839-byte/23-line resource receipt have SHA256
`8ef3d25ee3f074136460d8312dea8b6eed6a63516c75d302a8b997b899243fce`
and `686dcf652e537bfad448b7d2d95928855eed5201f56fe6105cfc58b0a20db9b9`.
Canonical H01 JSON/CSV regeneration is byte-identical at SHA256
`b85462e4b9dbc6fe03c6217bf11488604ed7d36306a081f2a84f2f3fb5b8cc4d`
and `10dbc70122b36493398622b9e5882e1508468a762fb4d5d6d9b5ee7e2ba3eb29`.
The PDF and LaTeX remain byte-identical to Turn2910, and all four pages passed
fresh visual review. Two independent recomputations and root review returned
GO. These checks qualify the package; they add no episode, request, inference,
label, confirmation result, physical-time evidence or gate advance.

The Turn2913 outcome-stratified fixed-index attrition extension was committed
as science commit `ee668d9252d3ca5ce85f76b25556bb2c2b03b603`; corrective layout commit
`8cf265835833d949ff1eeb22928f0f2af54c368d` removes the one added manuscript
sentence after the four-page gate rejected a five-page render. The resulting
net slice changes exactly five package paths by 6,484 insertions and nine
deletions; `paper/main.tex` and the PDF remain byte-identical to their preceding
qualified versions. A fresh detached build at `8cf26583` passed all **20
stages**, including **1073/1073 tests in 130.663 s**, in **146.85 s** wall time
with a peak resident set size of **920,520 KiB**. Its 564,094-byte/12,260-line
validation log, 568,709-byte/15,720-line build report, 569,457-byte/15,740-line
stdout and 1,267-byte/23-line resource receipt have SHA256
`c1a308567dd93f49f987c7b1d9d6cc47087a32fb01fb157d6b3a60843e0789e6`,
`268faed3991cd0c9e2069d3f69cb332dd52e88bf7c7fef7356183b4312ead368`,
`482f001b3ed4addc77e85e3b20c78648f209121eacaabad65a3c3134794a1ea3`
and `f8b12fd646c28d859b2d537f50ff3f6bce04f861d82b3124afb3e17704ccb8b6`.
The final serial protected-tree discovery passed **1080/1080 tests in 128.396
s** (**130.85 s** wall; peak resident set size **923,268 KiB**), including the
seven protected-only N3 methods. Its two-line/3,030-byte stdout and combined
1,115-line/141,847-byte verbose-test/resource stream have SHA256
`f714843bf3ee8fdae3dd813630b908f367dfffe4d4ec91e67fa0ebd802d028ce`
and `4261e843cf21c4ccb315745fa782aea0467315d6c181e9d6b42fbd793222af45`.

Canonical H01 v4 JSON is 41,741 lines/1,457,631 bytes at SHA256
`62ca025b0a5429bb798844c319cce9a1b45472f2692db034c67bc1a5d5995a0e`;
the 739-line/388,814-byte request CSV remains byte-identical at SHA256
`10dbc70122b36493398622b9e5882e1508468a762fb4d5d6d9b5ee7e2ba3eb29`.
All four freshly rendered pages were visually inspected and pass; PDF and TeX
remain `58a657b3ab86e36e6dfe6aa979efc9011c9c66700b6717385e66bc773998e965`
and `3abc21d3745781e2896e2834c6f7b62d464f82983f5fe8b3633767c620f0602d`.
Two initial suites run concurrently were discarded after a shared confirmation-
ledger mutation fixture caused the same raw-root identity race in both; the
three affected methods passed 3/3 serially. The first uncontaminated detached
build at `ee668d92` passed its 1,073 tests and all 20 stages but was also
discarded when the page-count gate found five pages. Neither discarded run is
promoted. Three independent read-only audits and root review returned GO on
the v4 identities, exact arrays and boundaries. The extension is retrospective
risk-set composition only: it estimates no outcome-stratum motion effect,
censoring independence, failure enrichment, language/model diagnosis,
forecast skill or new cohort. The 232-cell ceiling, 13 zero activity counters
and nine false gates remain unchanged.

The Turn2914 first-unavailability extension was committed as science commit
`949d22fcd793addae8920ac937d98e5fa95920c9`. It changes exactly six package
paths by 4,388 insertions and five deletions; `paper/main.tex`, the PDF and the
request CSV remain byte-identical. A fresh detached build at `949d22fc` passed
all **20 stages**, including **1074/1074 tests in 135.218 s**, in **151.43 s**
wall time with a peak resident set size of **921,432 KiB**. Its 564,222-byte/
12,261-line validation log, 713,396-byte/19,484-line build report,
714,144-byte/19,504-line stdout and 1,176-byte/23-line resource receipt have
SHA256 `5fe896bac56e5c3f41efff17d08dca6a831c736504218720e1433b0ff1872820`,
`96341b1e34a6b2d00672371a40dea0d21a22b698d5586b1c83b35b72c7e26f06`,
`a9bf4bc60bce799120f5f687b74535e3e845266fe49b06085edd7fc2c197737d`
and `a1bbe57bdb451d386c3a9b277a6164cf37a196cfd5946d34e18d5237ea078bae`.
The final serial protected-tree discovery passed **1081/1081 tests in 128.341
s** (**130.79 s** wall; peak resident set size **922,288 KiB**), including the
seven protected-only N3 methods. Its 144,166-byte/1,095-line stdout and
839-byte/23-line resource receipt have SHA256
`e66127d143a3dc5553b62b732720ac6420d8deac9eb0bd03c00cfbda51261f3c`
and `d5b77547e2c784b5114a3e4ff7e2b1d0e82ca363874d7df1dff42d4d81b4083a`.

Canonical H01 v5 JSON is 45,505 lines/1,594,790 bytes at SHA256
`c9f9159dee7b7cccef3c592e905d5e8a13a273f6bd545083ccbaaabfec766a63`;
the 739-line/388,814-byte request CSV remains at SHA256
`10dbc70122b36493398622b9e5882e1508468a762fb4d5d6d9b5ee7e2ba3eb29`.
The four-page PDF and LaTeX source remain byte-identical at the hashes recorded
above, and all four fresh renders passed visual review. Four independent
read-only audits plus root review returned GO on the recurrence identities,
limiter attribution, terminal evidence, mutation defenses and claim boundary.
These checks qualify a deterministic re-expression of existing v4 support;
they add no episode, request, inference, label, confirmation result,
independent scientific signal, activity or gate advance.

The Turn2915 post-snapshot H02 matched-entry supplement was introduced at
science commit `1ab8fa181d188f5b4b01fa2bb03fbf6fc237afae` and qualified at
`ea68f2df092c9a5b37009f889337c673069fa602`. Relative to Turn2914's
`c92cc5dcb0d8e2d32e156920161bf778195593da`, the final tree changes exactly
eight package paths by 1,752 insertions and four deletions. The supplement's
947-line/48,292-byte JSON and 28-line/3,934-byte CSV have SHA256
`95ad90a69aa6e0b5b2a135de6eb93a6a8d6290ec792bb19ba9d05e9864f96cb0`
and `1af3d5d2276f116bccda0a9ace3cbc657222bd98775be9ba39503a964346270a`;
both regenerate byte-exactly from the sealed H02 v1 package.

A fresh detached build at `ea68f2df` passed all **21 stages**, including
**1081/1081 tests in 142.522 s**, in **158.91 s** wall time with peak resident
set size **918,384 KiB**. Its 565,638-byte/12,291-line validation log,
724,555-byte/19,813-line build report, 725,340-byte/19,834-line stdout and
40-byte/one-line resource receipt have SHA256
`fe465074665f055fa242bcafc5e0537c912ba012553bb9a4641fdb09ec227715`,
`e84a038debde6f11415dcc8c4974c8950086bd4e3ccd3b62671288298edbb305`,
`19fba20e2e1ea3cc9661f5923fe8d0b310a75c9633f09adafac1b93f5b48af3e`
and `25b5386c9c235d838a149724a569435962d300cbe6d075a614c59fc5a5b6c578`.
The final post-documentation protected workspace discovery passed **1088/1088
tests in 128.151 s** (**130.59 s** wall; peak resident set size **922,544
KiB**); its
3,761-byte/12-line stdout and 40-byte/one-line resource receipt have SHA256
`295b6f59c146e236d6641ad171a75a91b70748222eff7aefa2e221f8ce232a5a`
and `38f058973f99e1838afc0cf96f515bfd5dad1b9b1678cae8e61cccaa5edb3506`.

The resulting 163,011-byte four-page PDF and 13,108-byte/106-line TeX source
have SHA256 `797757893cdc33f6b1b776943e01d3b5baffb75f80218f9b76c4224792b0a619`
and `596967f6e6392eb4b03a03cadffe7d4e510b87a8861e40ee01e50a1805be7c3f`.
There are no overfull boxes, unresolved citations/references or LaTeX errors;
all four fresh page renders passed visual review. Three independent read-only
audits plus root review returned GO on arithmetic, deterministic regeneration,
pre-execution source authentication, mutation defenses, capped-index semantics
and the compact manuscript boundary.

Nonqualifying intermediate receipts remain explicit: the first clean attempt
had one unrelated D1 fixture-identity failure that then passed five of five
isolated reruns; the next serial run passed the suite but correctly rejected a
five-page draft; and the first four-page run stopped on TeX's deterministic
`post-cap` to `postcap` extraction normalization. The final clean run above is
the only promoted build. These checks qualify post-result algebra over the same
108 retained episodes and 27 registered labels. They create no episode,
request, inference, label, confirmation result, independent scene/noise sample,
registered estimand, activity, gate advance or change to the 232-cell ceiling.

The Turn2916 retained H02 cap-composite decomposition was introduced at science
commit `ec86de756bfbf309546efa49474b3f8b8a23ea22`; compact four-page manuscript
corrections are `cdd07887f35eb84fd5d031801e7cfae6fb8eaa5a` and
`c96a535f9a212f81de7eb49c06839b3adba16db8`, and the fail-closed PDF boundary
checker correction is `1dfdf70e8c97533e02096fb84db93a60526c3202`. Relative to the
published Turn2915 seal `1622efadc689d95093789bef8d91d3e353d5ba3c`, that tree changes
exactly eight package paths by 1,734 insertions and three deletions. The
469-line/20,089-byte analyzer, 336-line/13,843-byte test, 808-line/37,325-byte
JSON and 28-line/4,581-byte CSV have SHA256
`bbfe43b7dde0ec4f30e7a7f2fb40c11389c8ced5175d6196dae512f7c363c11a`,
`99b0763dcd226dfbd5b2297399780638a14122c5d48498681de4160418ca7c74`,
`33d66ed365d2fe330f2a4988a578a6fdbdc83c335184fb9098aeaa0f0c676f09`
and `11fd4944b7cb5f9d3509bda6d9c0d1764d5d077461a05b027a13f980b16e3cb4`.
The eight focused tests pass and canonical regeneration is byte-exact.

A fresh detached authoritative build at `1dfdf70e` passed all **22 stages**,
including **1089/1089 tests in 130.367 s**, in **146.68 s** wall time with peak
resident set size **915,844 KiB**. Its 567,027-byte/12,316-line validation log,
736,752-byte/20,165-line build report, 737,576-byte/20,187-line stdout and
769-byte/23-line resource receipt have SHA256
`c373a2c16f4fe5dc117091ab0048416cfd2a83b92b98ffc4e15fd15d72803781`,
`01b32b25075f30f8f83fe5d0975a41427f2610b8b697079b57941f4f6a2c1f68`,
`31c98f81aa08064d3a22beaac2edd5db3c76818c194d50ba0f73c62dd7981ccc`
and `742aaf30c086f328fe1c6dcf17ee0a8686e57cd5676c72cea7588e2d4d0c3683`.
The final protected-tree discovery passed **1096/1096 tests in 128.188 s**
(**130.65 s** wall; peak resident set size **918,460 KiB**), including exactly
seven protected-only N3 methods. Its 145,909-byte/1,110-line stdout and
807-byte/23-line resource receipt have SHA256
`c0efdf3276250c236194decfa0349ec4aa6df99f1be27d0e183b6a297f11eeff`
and `7dd38247caceffdd9db9f736cd990eadaaf30f49cd6a27380391a86a952407ac`.

The generated 163,370-byte four-page PDF and 13,161-byte/106-line TeX source
have SHA256 `69fdc40e8a0c6eef1713bb92e085a04d9ffcca6e83a79a703219c8bc921eaba1`
and `e99c404f0eff8505b4bf25f1407fd66778f758d035117bb923df0b2f00d603f1`.
All four fresh page renders passed visual review with no clipping, overlap,
missing glyph, overfull box, unresolved citation/reference or LaTeX error.
Two preceding drafts were rejected at five pages, and the first four-page build
stopped fail-closed on the exact extracted boundary phrase; none was promoted.
Independent arithmetic and package reviews plus root review return GO. The
decomposition is exact post-result algebra: `Delta U = -450 Delta E + Delta Z`,
where `Z` is the zero-masked recorded-index term. Cross-layout status pairs are
not temporal transitions, entrant timing, causal or counterfactual effects,
survival estimates, physical time, forecast skill, D1/default evidence, A3--A9
evidence, a new cohort or a release gate. These software checks add no episode,
request, inference, annotation, label, confirmation or activity.

The Turn2917 retained H02 entry-status contribution ledger was introduced at
science commit `f844c8c5806b764f45a28c3e24f289b43b1e2f39`; compact manuscript
commits are `130fe429efe1baf3cf9f03d07f3616f25e2e3f5f`,
`bd8efe94ccc4afa0ded010da158eef316843ae5b` and
`ffa3215311043a09dc07c5f93caac096f8c0081b`. The fail-closed PDF-spacing
correction is `01373f90bbcffbb03a5ab2c057e1080b1d799f3a`, and
`4ca9105660cef772c335e4ad4407d5bec24f514e` restores and validates the sealed
540-descriptor inventory boundary. Relative to the published Turn2916 seal
`1ebf1d0404db5cfa6b874fb9f93ba7cee30e7ead`, the qualified tree changes exactly
eight package paths by 1,306 insertions and four deletions. The 483-line/
21,753-byte analyzer, 304-line/12,967-byte test, 386-line/15,060-byte JSON and
9-line/1,639-byte CSV have SHA256
`6aa3d0b1f6276ac57b8b3838f2ac22bae5929a5d764f57a5370cb75ba04e8a5e`,
`ac35dfcb13f07f4760faa14358cbb1ab577ca55a03bc76b6a613dc2e20830d40`,
`9bc4c8ade7cdcea065a7eee53d74059c0098ca79a55ddf9f930f29c9d771d9d9`
and `5a48495fb332ca7ec764df36f69e064899797cea562913f1add5840b3923e1ba`.
Canonical regeneration is byte-exact; 32 focused and adjacent H02 tests pass.

A fresh detached authoritative build at `4ca91056` passed all **23 stages**,
including **1099/1099 tests in 135.458 s**, in **151.86 s** wall time with peak
resident set size **928,468 KiB**. Its 570,003-byte/12,374-line validation log,
752,799-byte/20,556-line build report, 753,662-byte/20,579-line stdout and
833-byte/23-line resource receipt have SHA256
`cdcb3e7f7de9e0976e06c91e5b7b3623f78cb54a0ef342d6ee1819ed1e94cd92`,
`19e46f6673cf21c0220963c9dd9ecfbd0852dd43945a2a973f9e183870593795`,
`01ec9ccacb994caff67867113337918ccdaddefee5811b4a01b8513c83b3315b`
and `9f78cb9391b11cc7bbcb72f164515804ca90aef5c13fc9e487f1dcc94e277c7d`.
The protected-tree discovery passed **1106/1106 tests in 128.759 s**
(**131.21 s** wall; peak resident set size **919,340 KiB**), including exactly
seven protected-only N3 methods. Its 147,162-byte/1,120-line stdout and
839-byte/23-line resource receipt have SHA256
`d1c766fd589d85841f3041f14c10f09258e6f804517054ce815e4ddf8a97c73e`
and `1cfcdd67a633a70854c61008cff0ec587a77f449f5a71e98726569dcfcbb6304`.
The protected 23-path/1,626-byte status stream remained byte-identical before
and after the suite.

The generated 163,473-byte four-page PDF and 13,152-byte/106-line TeX source
have SHA256 `e754568e3f92ac2f903c09d3d30939e527a27c421d7fe8064b8ecba0297594a2`
and `92ac3803b35de6a75fc17f0df9d5b9bb317f0943b494736456f86099e9c83945`.
All four fresh page renders passed visual review with no clipping, overlap,
missing glyph, overfull box, unresolved citation/reference or LaTeX error.
Three full detached attempts were made and only this one is promoted: the first
correctly rejected a five-page draft, and the second stopped at one regression
failure among 1,099 tests after compaction removed the required `540` marker.
An independent audit also caught pdfTeX extraction of `+227` as `+ 227`; the
validator now accepts only that whitespace normalization and all 66 required
markers pass. Three independent read-only reviews plus root review return GO.

The eight ledger rows exactly partition the already sealed command totals.
LEFT contributes `0`, `-9900+3008=-6892`, `450-223=227` and `-100`; RIGHT
contributes `0`, `-8550+2829=-5721`, `0` and `-382`. Three empty strata retain
`conditional_mean=null`; empty sums are not observed zero effects. The combined
ledger closes `-18000+5132=-12868=2*(-6434)`. These are outcome-defined
cross-layout comparison categories, not transitions, principal strata,
mediators, conditional/causal effects, entrant timing, survival, physical time,
forecast skill, D1/default evidence, A3--A9 evidence, a new cohort or a gate.
The checks create no episode, request, inference, annotation, label,
confirmation, activity or change to the 232-cell ceiling.

Figure and PDF subprocesses receive `SOURCE_DATE_EPOCH=1786718341` for the
retained study-source commit and `FORCE_SOURCE_DATE=1`, and the build report
records both values. The retained PDF still has TeX-produced wall-clock
metadata, so these settings do not establish byte-identical PDF rebuilding.
The pinned TeX Live 2026 latexmk 4.88 and pdfTeX 1.40.29 toolchain produced both
the checked-in manuscript and the clean isolated four-page retained-analysis package with
the exact PDF SHA256 recorded above. The build had zero errors, overfull boxes,
undefined citations or undefined references. All four isolated pages were
rendered and passed visual review.

`results/validation.log` and `results/build_report.json` are generated build
products, not retained checkout evidence in this protected worktree. The final
authoritative isolated validation log is 562,124 bytes/12,245 lines with
SHA256 `b8b67b85ae76bcb8bdd1b8141503022b87f1b17d955b808d1036629112896069`;
its 170,490-byte/5,004-line build report has SHA256
`3a342066e80cb64ae1b89c3d54dcf5bc2ff27e133b09ffedc472679823f47618`.
The complete 171,194-byte/5,023-line build stdout and 1,210-byte/23-line resource
receipt have SHA256
`da45ed19c5a5d9146d779585878aba946480125ba2643f2542ce8795856d419d` and
`9750548454f5a3299849ee0080b62bf44c9784668f8944ed2c79d1c860f7796b`.
The post-merge protected discovery passed **1067/1067 tests in 128.005 s**
(**130.43 s** wall; peak resident set size **910,392 KiB**), including exactly
seven protected-only N3 methods. Its 142,409-byte/1,081-line stdout and
869-byte/23-line resource receipt have SHA256
`996687e510658f5137c91cbcbc970d697afb510c96ad85c110cd10646aed88a4` and
`2bebd163e0253099a2c8a7e34ed77077696fdb3b33e5ca64e842559cd682babf`.
Independent inspection confirmed that the report binds the exact environment,
all 19 successful stages, the four-page 162,589-byte PDF and 12,767-byte LaTeX
source above, all nine grouped retained-analysis packages and all 13 registered
historical IDs, and all current activity counters/gates at zero/false. The
coverage review returned GO after pinned-source authentication, mutation tests,
canonical regeneration and rendered-page inspection. Durable
inputs and receipts are checked in under `paper/data/` and `paper/media/`; the
execution status/progress ledger records this qualification receipt without
recreating the protected deleted results.

The extractor verifies nine source Git blobs at upstream commit
`ce561e66f82e95055e39d3d7711691982f6b2086`. It recomputes counts and continuous
measurements from episode rows and reconciles them against the source summaries.
It extracts three separate 108-episode cohorts; the two position-reflection
cohorts, 216 episodes total, provide the primary manuscript experiment. The
separate DreamZero symmetry experiment is not pooled with either reflection
experiment. H05 is separately authenticated from its registered report and is
not pooled into those trial counts.

An independent reviewer checked all eight success counts, the instruction-response
and placement-depth means, intervals and measurement definitions against the pinned
episode rows. The final manuscript incorporates the following corrections:

- Instruction response compares final cube–bowl offsets, not absolute cube endpoints.
- The success cone extends 45 degrees on either side of the requested direction.
- Continuous position measurements include failed trials.
- The commands and 450-action limit are stated in the experimental setup.
- For the primary position-reflection cohorts, DreamZero results are descriptive
  because its effective model-noise seed is fixed; only Nano receives bootstrap
  confidence intervals in the primary paper figure. The separate preregistered
  retained analyses report matched-block intervals over their registered seed
  blocks; they do not create independent model-noise draws.
- DreamZero's original 41 failures are described by the recorded execution checks:
  26 pickup, 14 transport and one wrong-side placement. These are not diagnoses
  of language understanding.
- The H01 requested-region diagnostic keeps four reflected-RIGHT no-entry traces
  cap-censored, reports 104 entries and the matched entry-index interaction, and
  does not turn action indices into physical time or goal-terminated transients
  into common-horizon sustained entries.
- The retained H01 request-window package authenticates 108 cells and 738
  request descriptors. Exact start-to-start-plus-32-action windows exist for
  632 requests; 106 targets are censored by 100 goal-stopped successes and six
  cap-censored failures, with 1,748 observed prefix actions across those
  censored windows. Equal-cell cube displacement means are 0.0450153,
  0.0785662, 0.0765150 and 0.0680017 m across the four registered conditions.
  Descriptor path/hash identities are unique, but forecast payload bytes were
  not reopened, reverified or scored. This is retrospective execution-motion
  and descriptor-provenance evidence, not physical-time pairing, an A6
  threshold, generated-versus-executed comparison, forecast accuracy, A3--A9
  completion, new inference, new labeling or a change to the 232-cell core.
  The matched complete-window supplement retains 114 control and 113 reflected
  exact LEFT/RIGHT pairs and exposes 101 and 77 otherwise-complete records that
  lack an exact counterpart endpoint. Of those exclusions, 43 counterpart
  requests terminate before the 32-action endpoint and 135 owner requests
  begin at or after the counterpart episode ended. Equal-seed signed means are
  positive/tied/negative for 26/1/0 control and 27/0/0 reflected seed labels,
  but raw window signs are mixed, every first 0--32 window is tied, and the
  same-request-index reflected-minus-control audit is only 15/11/1
  positive/negative/tied. These telescoping, outcome-selected action-window
  summaries are not independent endpoints, physical-time rates, movement
  thresholds, instruction-understanding results or forecast scores.
  The fixed request-index extension preserves all 27 paired seeds per layout
  through actions 32--64 (signs 2/6/19 and 3/8/16), then retains 26 and 27 at
  actions 64--96 (20/6/0 and 25/1/1). Its exact four-condition support is 26
  seeds at 64--96 with interaction signs 15/11/0, falls to 12 at 96--128 and
  two at 128--192, and is empty later. Full-27-seed strict-positive bounds keep
  every unavailable seed unresolved rather than carrying a stopped trace.
  Request indices 10--14 remain explicit zero-support rows rather than being
  omitted. This is same-population descriptive selection accounting, not a new
  confirmatory estimand or cohort, physical-time endpoint, A8 completion,
  activity/gate advance or change to the 232-cell core.
  The v5 first-unavailability decomposition closes every support curve against
  its exact first losses. Control losses by index are
  0,0,1,8,10,3,4,0,0,1,0,0,0,0,0 and reflected losses are
  0,0,0,10,12,1,2,0,1,0,1,0,0,0,0. LEFT-only/both/RIGHT-only limiters are
  13/3/11 and 19/6/2; all 63 per-layout limiting-cell appearances, including
  the limiters tied in nine pairs, are success-coded, with 61 partial
  goal-stopped requests and two control-LEFT episodes ending exactly at
  action 128. Four-condition first losses are 1,14,10,2 at indices 2,3,4,6.
  First-loss arrays are discrete differences of the existing support curves;
  only limiter identities and terminal forms are newly exposed, not an
  independent endpoint, estimand, cohort or scientific signal. Historical
  successes goal-stop while failures remain observed to the 450-action cap,
  and no pair contains two failures. That stopping structure mechanically
  yields zero cap-failure limiters; it is not an outcome association and does
  not establish censoring independence, failure enrichment, outcome-stratified
  motion effects, physical time, forecast skill, language/model diagnosis,
  activity or a gate.
- The retained H02 package byte-authenticates nine compact Git objects
  (831,259 bytes/1,429 lines), including the bridge that defines the 45-degree
  cone and three-sample sustained rule, and rejoins the exact 108-cell/54-pair V3-B003
  population and reproduces 13/54 original-layout versus 50/54 reflected-layout
  successes, the five-category failure inventory and registered endpoint,
  requested-depth and binary-success interactions. The compact requested-cone
  fields add a post-result descriptive 13/54 versus 53/54 recorded-entry
  summary: 63 entries have an observed three-sample run and three successful
  goal-terminated entries have no observed three-sample follow-up; they are not
  classified transient exits. All 42 no-entry traces remain action-450
  cap-censored failures. These are upstream action indices, not physical time or
  a raw-XYZ replay. The canonical 27-row matched-block CSV exposes the four
  cells and entry/censor contrasts per seed. A separate post-snapshot supplement
  preserves that sealed v1 package and weights LEFT and RIGHT equally inside
  every registered label: the reflected-minus-control descriptive contrasts
  are +0.740741 for the recorded requested-cone-entry indicator (25/2/0
  positive/zero/negative blocks) and -238.296 action indices for the capped
  first-entry-index variable $U$ (median -267.5, range -328.5 to -37.5;
  negative in 27/27 blocks). LEFT/RIGHT $U$ contrasts are -250.556/-226.037 and
  the command-by-layout difference-in-differences is +24.519 with
  positive/zero/negative=19/0/8. $U$ is the recorded first-entry index, or 450
  when no entry was recorded by the common administrative cap. $U=450$ is not
  an event at 450 or an imputed entry: this composite mixes entry-by-cap
  incidence with first-entry index among entrants, so it is not a pure timing
  or physical-latency contrast and does not extrapolate event times beyond the
  cap. These are post-result aggregates of the same exported rows, not a
  model-based survival curve or hazard, censoring-independence claim, new
  registered estimand, cohort, independent scene/noise sample, equivalence
  result or p-value claim. A standalone exact decomposition authenticates and
  regenerates that supplement, then applies $U=450(1-E)+Z$ within every row,
  where $Z$ is the recorded entry index when entry is observed and algebraic
  zero otherwise.
  The block-balanced cap-incidence term -333.333 plus
  zero-masked recorded-index term +95.037 closes to -238.296; sums close as
  -9000 + 2566 = -6434, with LEFT/RIGHT/interaction closures
  -9450 + 2685 = -6765, -8550 + 2447 = -6103 and 900 - 238 = 662.
  LEFT cross-layout entry-status pairs are 22/1/4/0 and RIGHT 19/0/8/0 for
  control-no-entry/reflected-entry, control-entry/reflected-no-entry,
  control-entry/reflected-entry and neither-entry. These are comparison
  categories whose exact contribution ledger partitions the same totals:
  LEFT closes through nonempty status rows as -9,900 + 3,008 = -6892,
  450 - 223 = 227 and 0 - 100 = -100; RIGHT closes as
  -8,550 + 2,829 = -5721 and 0 - 382 = -382.
  41 control-no-entry/reflected-entry command-label comparisons contribute
  -18,450 + 5,837 = -12,613; 12 both-entry comparisons contribute
  0 - 482 = -482; and one control-entry/reflected-no-entry comparison
  contributes 450 - 223 = 227. Three of eight command/status rows are empty,
  with no conditional mean. The ledger total -18,000 + 5,132 = -12,868 is
  exactly twice the block-balanced -6,434 sum. These are comparison
  categories across different layouts, not temporal transitions,
  within-episode conversions, principal strata or counterfactual changes. The $Z$ term is
  a zero-masked recorded-index term, not an entrant-only timing effect; its zero
  under no entry is algebra, not an imputed entry index. This remains
  deterministic post-result algebra, not causal mediation, physical latency,
  survival inference, an independent sample, a registered estimand or a gate
  advance. Its 540 distinct raw
  path/SHA descriptors total 237,163,185 declared bytes; the external PVC bytes
  were not opened or rehashed. H02 remains custom DreamZero negative-branch
  action guidance s2 with fixed effective noise 1140, not default DreamZero or
  official conditional D1. No forecast-accuracy, current raw recovery, new
  inference, annotation, confirmation, release gate or 232-cell-bound change
  follows; all current activity counters are zero and all gates false.
- The retained H03/H04 package byte-authenticates seven compact Git objects
  (810,860 bytes/18,939 lines) and rejoins six baseline/intervention cells for
  each model. Nano changes from g3 6/6 to g1 4/6; DreamZero changes from the
  conditional-action-equivalent s1 baseline 3/6 to custom negative-branch s2
  4/6. All six paired-seed records independently reproduce and source-validate
  endpoint ordering, separation and right-minus-left requested-margin gaps; all
  three command pairs remain endpoint ordered per configuration, but
  every cell shares one retained reset fingerprint and DreamZero fixes effective
  model-noise seed 1140. The pilots are post-result, exploratory and descriptive;
  their historical baselines are not double counted, the models are not pooled,
  and H04's historical custom-s2 arm neither selects a new-confirmation D2
  extension nor qualifies D1. Original raw files remain external
  and were not replayed. No new inference, episode, annotation, confirmation,
  release gate or change to the 232-cell core follows.
- The retained H05 package authenticates a 15-seed by seven-level by two-command
  design, reproduces the 1.125 m/m registered slope and its 20,000-resample
  matched-seed interval, and preserves the nonmonotone level means and
  out-of-support fitted crossing. Its 315 raw-source identities are retained in
  the manifest, but the raw bytes are outside this checkout and were not
  independently rehashed here. The 210 episodes and 1,423 requests are
  authenticated historical-population counts, not current-turn activity; the
  H05 receipt records all 13 current activity counters as zero and all nine
  release gates as false.
- The retained H06 package authenticates the prospectively registered
  V3-E004 compact tables and independently rejoins the 216-episode/108-pair
  inventory-matched Nano grid. It reproduces the mean within-seed depth-gap
  slope of 0.020 m per unit realised asymmetry, its 20,000-resample interval
  and exact sign test. Level zero remains excluded because its object inventory
  differs. Compact Git rows are rehashed, but original raw simulator files are
  external; H06 is not independent-scene, position-reflection, equivalence,
  annotation, forecast accuracy or release-gate evidence. Its receipt likewise
  records all 13 activity counters as zero and all nine gates as false.
- The retained H07/H08 endpoint package independently rejoins 108 episodes and
  54 LEFT/RIGHT pairs for each of DreamZero-s2 and Cosmos Edge over the exact
  27-seed, two-level registered grids. It reproduces the registered
  $s=1$-minus-$s=0$ depth, binary-success and endpoint-redirection
  interactions, their 20,000-resample intervals and exact layout-label
  permutations. The level-zero inventory omits companion objects, so these are
  separate inventory-transition contexts rather than position-reflection
  replications. Compact rows are rehashed; original raw simulator files remain
  external. No equivalence, forecast-accuracy, annotation, confirmation or
  release-gate claim follows, and all current activity/gate counters remain
  zero/false.
- The retained H09/H10 wording package byte-authenticates the frozen 480-row
  three-checkpoint V3-C001 registration and all 320 ABLATION_SPEC-selected
  Nano/Edge compact episode rows. It reconstructs 120 same-seed direct-control
  contrast rows without pooling models or exporting raw PVC paths; the excluded
  GR00T cohort remains explicitly disclosed. Nano/Edge successes are 134/160
  and 123/160; historical requests/failures are 1,234/26 and 1,365/37. The
  requested-margin and endpoint-shift views are algebraically paired rather
  than independent signals. Confirmatory wording inference was not registered,
  so all contrasts remain exploratory descriptive repeated-block evidence. No new
  inference, episode, annotation, confirmation or release gate follows, and the
  232-cell core bound remains unchanged.
- The retained H11/H12 Phase-A package byte-authenticates the exact 54
  authorized-new direct-command cells and 27 matched LEFT/RIGHT reset blocks
  for each model while excluding six preserved V2 cells per model. Its eight
  payload objects total 5,636,174 bytes; two design/overlap helpers bring the
  authenticated boundary to ten objects and 5,654,967 bytes. Nano
  succeeds in 51/54 and reconciles 349 decoded RGB futures/action chunks;
  DreamZero-s2 succeeds in 3/27 LEFT and 17/27 RIGHT and reconciles 2,554
  latent futures but only 54 official full-reset decodes and 54 rollout videos.
  The models and interfaces remain separate, fixed model-noise seed 1140 is not
  an independent draw, and H12's numerically matching H07 directional totals
  arise from identity-distinct seed cohorts. Original raw Phase-A payload bytes
  remain external and were not revalidated. No raw recovery, forecast accuracy,
  annotation, confirmation, new activity, gate advance or change to the
  232-cell core bound follows.
- The retained H13 evaluator-development package byte-authenticates five audit
  objects (998,249 bytes) and reopens all 761 ledger-listed pinned upstream
  source Git blobs (7,230,812 bytes; 752 cache files). It strictly validates
  immutable reconciled rows and matched-block identities for 80 episodes,
  752 replan chunks, 3,008 cached forecast frames and 20 four-episode blocks,
  then independently replays the registered arithmetic.
  The historical scorer labels 421/752 chunks but covers only 25/97
  executed-positive horizons. Its same-covered-subset apparent historical-label
  agreement is 413/421 versus 396/421 for constant-negative; 331 abstentions
  imply historical-label missingness bounds of 413/752--744/752, not a fidelity
  interval. The endpoint/threshold and 2,000-resample matched-block summaries are
  descriptive evaluator diagnostics only. A frozen 160-row stratified sampling
  design reproduces exactly, but verified media/alignment items and collected
  human labels remain 0/0/0. Corrected fidelity remains blocked on 840 required
  configuration/alignment/trajectory paths plus current raw-media recovery/replay. No new
  model inference, behavioral episode, fidelity estimate, annotation or
  confirmation gate follows: all 13 activity counters are zero, all nine gates
  false and the 232-cell core ceiling unchanged.
- The retained historical-evidence coverage audit authenticates 50 exact Git
  objects (2,054,896 bytes/45,726 lines; canonical manifest SHA256
  `26dac013bb654e28145e0f2bdfbfe2d8622080c9af84b3773efe5c217a337b38`)
  at the pinned Turn2908 control snapshot. It maps every registered H01--H13 ID
  exactly once to nine grouped analysis packages with explicit closed claim
  boundaries. H01, H02 and H07 have additional primary, supplemental or
  presentation roles that do not create extra cohorts. H02, H04, H07 and H12
  remain historical custom-s2 configurations, not official conditional D1.
  Episode counts and models are not pooled; no outcome is re-estimated. Raw
  H01/H02 recovery, manifest-host availability, physical-time pairing,
  independent labels, qualified D1, prospective varied-layout forecast
  accuracy and fixed-duration H01 evidence remain unestablished. The audit
  creates no new inference, behavioral episode, label, generalization,
  equivalence, language-understanding or forecast-accuracy evidence, and changes
  neither the 232-cell bound nor any science/release gate.
- The four representative seed-9400 videos are authenticated and fully decoded,
  but remain presentation media rather than blinded annotation inputs, timing
  maps, forecast-accuracy evidence or a release-gate advance.

### Turn2918 exact H02 one-label-deletion sensitivity

The retained package authenticates and byte-regenerates the sealed Turn2917
entry-status contribution inputs before dynamically executing them. It deletes
each of the 27 registered environment labels once, leaving 26 labels, and
reports 324 exact rational deletion means across four scopes and the three
unchanged components of $U=450(1-E)+Z$. All 12 means preserve their
full-population signs in all 27 deletions. Composite $U$ ranges are LEFT
-268.923 to -247.423, RIGHT -234.231 to -221.923, block-balanced -246.019 to
-234.827 and command-by-layout interaction +13.577 to +45.808 action indices.
All deletion-level component identities and balanced/interaction scope
identities close exactly; tied extrema are retained rather than discarded.

This is deterministic post-result sensitivity for the fixed observed
27-label population. It is not jackknife inference, a standard error,
confidence interval, p-value, bootstrap or other resampling distribution,
influence-function analysis, equivalence evidence, robustness to unobserved
labels, independent-scene/model-noise generalization, physical time,
entrant-only timing, a registered estimand or a new cohort. The analysis adds
zero episodes, requests, labels, annotations, retrievals or inference, retains
all 13 activity counters at zero and all nine gates false, and does not change
the 232-cell core bound. Authoritative build, protected-tree and four-page PDF
qualification receipts are recorded below only after those checks complete.

## Remaining research and submission work

The main results cover two checkpoints, one task and two fixed physical layouts;
H03/H04 reuse one retained physical reset across three seed labels and two
commands per configuration; they are not independent scenes or confirmatory
guidance evidence. H05 adds seven positions within one fixed simulator setup,
not seven scenes.
H06 adds four inventory-matched positive symmetry levels within 27 repeated
seed blocks, not 216 independent scenes. H07/H08 add two endpoint-only model
contexts over repeated seed blocks; their level-zero inventory differs from
level one. H09/H10 reuse 20 seeds across eight wording/direction cells per model
in one neutral scene; they are not 320 independent scenes or confirmatory
wording evidence. H11/H12 each reuse 27 neutral-scene reset labels across two
directions; they are separate model-specific development/recovery cohorts, not
54 independent scenes, a cross-model experiment or recovered raw evidence.
H13 is an older evaluator audit with missing raw alignment/media prerequisites,
not an 80-scene behavioral study, a fidelity estimate or annotation completion.
They establish neither prediction accuracy nor the causal benefit of world modeling.
They also do not isolate visibility, reachability or grasp geometry as the mechanism.

`docs/NEXT_EXPERIMENT.md` specifies an eight-trial recording check followed by a
comparison of generated futures with actual outcomes at matching times. It has not
been launched. New layouts, qualified model-noise sampling and independent image
labels remain future work. No new model inference, remote compute or human labeling
was performed during the rewrite.

Final authorship, publication eligibility and the workshop's final submission rules
remain to be confirmed before submission. No submission or public release occurred.
