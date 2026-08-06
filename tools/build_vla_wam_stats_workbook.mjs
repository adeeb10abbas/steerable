#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoRoot = path.resolve(process.argv[2] ?? process.cwd());
const outputDir = path.resolve(
  process.argv[3] ?? path.join(repoRoot, "outputs/vla_wam_research_handoff"),
);
const previewDir = path.resolve(
  process.argv[4] ?? path.join("/private/tmp", "vla_wam_stats_previews"),
);

const comparisonPath = path.join(
  repoRoot,
  "artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison.json",
);
const catalogPath = path.join(
  repoRoot,
  "artifacts/vla_wam_shared_v2/media/media_catalog.json",
);
const galleryPath = path.join(
  repoRoot,
  "artifacts/vla_wam_shared_v2/media/video_first_gallery_manifest.json",
);

const comparison = JSON.parse(await fs.readFile(comparisonPath, "utf8"));
const catalog = JSON.parse(await fs.readFile(catalogPath, "utf8"));
const gallery = JSON.parse(await fs.readFile(galleryPath, "utf8"));

const workbook = Workbook.create();
const dashboard = workbook.worksheets.add("Dashboard");
const results = workbook.worksheets.add("Behavioral Results");
const media = workbook.worksheets.add("Media Inventory");
const probes = workbook.worksheets.add("Interface Probes");
const definitions = workbook.worksheets.add("Definitions & Sources");

const COLORS = {
  navy: "#0B1F33",
  teal: "#0F766E",
  blue: "#2563EB",
  orange: "#EA580C",
  purple: "#7C3AED",
  green: "#15803D",
  paleBlue: "#E8F1FB",
  paleTeal: "#E6F4F1",
  paleOrange: "#FFF1E7",
  paleGray: "#F3F6F8",
  border: "#D7E0E8",
  text: "#172033",
  muted: "#526076",
  white: "#FFFFFF",
};

function setTitle(sheet, range, title, subtitle) {
  sheet.getRange(range).merge();
  const first = range.split(":")[0];
  sheet.getRange(first).values = [[title]];
  sheet.getRange(range).format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, size: 20 },
    verticalAlignment: "center",
  };
  const [col, rowText] = first.match(/([A-Z]+)(\d+)/).slice(1);
  const row = Number(rowText);
  const endCol = range.split(":")[1].match(/[A-Z]+/)[0];
  const subtitleRange = `${col}${row + 2}:${endCol}${row + 3}`;
  sheet.getRange(subtitleRange).merge();
  sheet.getRange(`${col}${row + 2}`).values = [[subtitle]];
  sheet.getRange(subtitleRange).format = {
    fill: COLORS.paleBlue,
    font: { color: COLORS.text, italic: true, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
}

function styleHeader(range) {
  range.format = {
    fill: COLORS.teal,
    font: { bold: true, color: COLORS.white },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: COLORS.border },
  };
}

function styleBody(range) {
  range.format = {
    font: { color: COLORS.text, size: 10 },
    verticalAlignment: "top",
    borders: { preset: "all", style: "thin", color: COLORS.border },
  };
}

for (const sheet of [dashboard, results, media, probes, definitions]) {
  sheet.showGridLines = false;
}

// Behavioral results ---------------------------------------------------------
setTitle(
  results,
  "A1:X2",
  "Behavioral results — exact counts and formula-derived rates",
  "DROID/RoboLab and RoboTwin are separate arenas. Infrastructure-invalid attempts are excluded; missing metrics remain blank, never zero.",
);

const resultHeaders = [
  "Arena", "Class", "Model", "Valid episodes", "LEFT successes", "LEFT trials",
  "LEFT rate", "RIGHT successes", "RIGHT trials", "RIGHT rate", "Total successes",
  "Total trials", "Overall success", "Aligned pairs", "Endpoint pairs", "Alignment rate",
  "Distinct-action pairs", "Action pairs", "Distinct-action rate", "Invalid attempts",
  "Future evidence status", "Model ID", "Hash-bearing evidence sources", "Claim boundary",
];
results.getRange("A6:X6").values = [resultHeaders];
styleHeader(results.getRange("A6:X6"));

const resultRows = comparison.rows.map((row) => [
  row.arena,
  row.model_class,
  row.model,
  row.valid_n,
  row.left_success.count,
  row.left_success.trials,
  null,
  row.right_success.count,
  row.right_success.trials,
  null,
  null,
  null,
  null,
  row.paired_endpoint_alignment.count,
  row.paired_endpoint_alignment.trials,
  null,
  row.paired_action_distinctness.count,
  row.paired_action_distinctness.trials,
  null,
  row.invalid_attempt_count,
  row.future_evidence_status,
  row.model_id,
  row.sources.map((source) => `${source.path}@sha256:${source.sha256}`).join("; "),
  row.claim_boundary,
]);
results.getRange(`A7:X${6 + resultRows.length}`).values = resultRows;
styleBody(results.getRange(`A7:X${6 + resultRows.length}`));

for (let index = 0; index < resultRows.length; index += 1) {
  const row = 7 + index;
  results.getRange(`G${row}`).formulas = [[`=IFERROR(E${row}/F${row},"")`]];
  results.getRange(`J${row}`).formulas = [[`=IFERROR(H${row}/I${row},"")`]];
  results.getRange(`K${row}`).formulas = [[`=E${row}+H${row}`]];
  results.getRange(`L${row}`).formulas = [[`=F${row}+I${row}`]];
  results.getRange(`M${row}`).formulas = [[`=IFERROR(K${row}/L${row},"")`]];
  results.getRange(`P${row}`).formulas = [[`=IFERROR(N${row}/O${row},"")`]];
  results.getRange(`S${row}`).formulas = [[`=IFERROR(Q${row}/R${row},"")`]];
}
results.getRange(`G7:G${6 + resultRows.length}`).format.numberFormat = "0.0%";
results.getRange(`J7:J${6 + resultRows.length}`).format.numberFormat = "0.0%";
results.getRange(`M7:M${6 + resultRows.length}`).format.numberFormat = "0.0%";
results.getRange(`P7:P${6 + resultRows.length}`).format.numberFormat = "0.0%";
results.getRange(`S7:S${6 + resultRows.length}`).format.numberFormat = "0.0%";
results.tables.add(`A6:X${6 + resultRows.length}`, true, "BehavioralResultsTable");
results.freezePanes.freezeRows(6);
results.freezePanes.freezeColumns(3);
results.getRange("A:A").format.columnWidth = 28;
results.getRange("B:B").format.columnWidth = 10;
results.getRange("C:C").format.columnWidth = 36;
results.getRange("D:T").format.columnWidth = 13;
results.getRange("U:U").format.columnWidth = 32;
results.getRange("V:V").format.columnWidth = 30;
results.getRange("W:X").format.columnWidth = 58;
results.getRange(`A7:X${6 + resultRows.length}`).format.wrapText = true;
results.getRange(`M7:M${6 + resultRows.length}`).conditionalFormats.add("colorScale", {
  colors: ["#FEE2E2", "#FEF3C7", "#DCFCE7"],
  thresholds: ["min", "50%", "max"],
});
results.getRange(`P7:P${6 + resultRows.length}`).conditionalFormats.add("dataBar", {
  color: COLORS.teal,
  thresholds: ["min", "max"],
});

// Media inventory ------------------------------------------------------------
setTitle(
  media,
  "A1:K2",
  "Video inventory — execution, prediction, and support media",
  "One row per committed MP4. Canonical gallery items are the reading order; archives, alternate encodes, and reconstruction parts are not additional episodes.",
);
const mediaHeaders = [
  "Class", "Model", "Arena", "Evidence kind", "Publication role", "Canonical?",
  "Repository path", "Size (MiB)", "Bytes", "SHA-256", "Source manifest",
];
media.getRange("A6:K6").values = [mediaHeaders];
styleHeader(media.getRange("A6:K6"));
const mediaRows = catalog.videos.map((row) => [
  row.model_class, row.model, row.arena, row.evidence_kind, row.publication_role,
  null, row.path, null, row.bytes, row.sha256, row.source_manifest,
]);
media.getRange(`A7:K${6 + mediaRows.length}`).values = mediaRows;
styleBody(media.getRange(`A7:K${6 + mediaRows.length}`));
for (let index = 0; index < mediaRows.length; index += 1) {
  const row = 7 + index;
  media.getRange(`F${row}`).formulas = [[
    `=IF(OR(E${row}="canonical gallery",E${row}="canonical paired comparison"),"Yes","No")`,
  ]];
  media.getRange(`H${row}`).formulas = [[`=I${row}/1048576`]];
}
media.getRange(`H7:H${6 + mediaRows.length}`).format.numberFormat = "0.00";
media.tables.add(`A6:K${6 + mediaRows.length}`, true, "MediaInventoryTable");
media.freezePanes.freezeRows(6);
media.freezePanes.freezeColumns(2);
media.getRange("A:A").format.columnWidth = 10;
media.getRange("B:B").format.columnWidth = 38;
media.getRange("C:C").format.columnWidth = 32;
media.getRange("D:E").format.columnWidth = 28;
media.getRange("F:F").format.columnWidth = 11;
media.getRange("G:G").format.columnWidth = 76;
media.getRange("H:I").format.columnWidth = 14;
media.getRange("J:J").format.columnWidth = 68;
media.getRange("K:K").format.columnWidth = 72;
media.getRange(`A7:K${6 + mediaRows.length}`).format.wrapText = true;
media.getRange(`D7:D${6 + mediaRows.length}`).conditionalFormats.add(
  "containsText",
  { text: "actual rollout", format: { fill: COLORS.paleTeal, font: { color: COLORS.green } } },
);
media.getRange(`D7:D${6 + mediaRows.length}`).conditionalFormats.add(
  "containsText",
  { text: "prediction", format: { fill: COLORS.paleOrange, font: { color: COLORS.orange } } },
);

// Nonbehavioral interface probes ---------------------------------------------
setTitle(
  probes,
  "A1:J2",
  "Nonbehavioral and release-gate evidence",
  "These rows have zero behavioral episodes and never enter success denominators.",
);
const probeHeaders = [
  "Model", "Class", "Scope", "Behavioral episodes", "Evidence type", "Gate result",
  "Execution video", "Prediction video", "Why no success score", "Primary source",
];
probes.getRange("A6:J6").values = [probeHeaders];
styleHeader(probes.getRange("A6:J6"));
const probeRows = [
  [
    "Cosmos3 Edge base — DROID", "WAM", "Fixed-observation interface probe", 0,
    "Generated future + unexecuted actions", "Repeat deterministic; LEFT/RIGHT changed",
    "Unavailable", "Available (canonical)",
    "Exact Franka + Robotiq CuRobo control mapping was not verified.",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_base_v2a013_fixed_observation.json",
  ],
  [
    "Cosmos3-Super base", "WAM", "Image-only interface probe", 0,
    "Generated future + unexecuted actions", "Repeat deterministic; LEFT/RIGHT changed",
    "Unavailable", "Available (canonical)",
    "Authorized arm was image-only: no state, simulator, or controller execution.",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_super_image_only_v2a014_result.json",
  ],
  [
    "Cosmos-Reason2 2B", "Diagnostic", "Static reasoning diagnostic", 0,
    "Reasoning output only", "Not a robot policy or behavioral gate",
    "Unavailable", "Unavailable",
    "The retained arm does not emit executable policy actions for this protocol.",
    "artifacts/vla_wam_shared_v2/pilot/expansion/cosmos_reason2_2b_readiness.json",
  ],
  [
    "π0-FAST current-stack wording replication", "VLA release gate",
    "Fixed-observation action-sensitivity gate", 0, "Action tensor diagnostic",
    "Failed: LEFT and RIGHT action tensors were identical", "Unavailable", "Not applicable",
    "The preregistered release gate stopped before behavioral episodes.",
    "artifacts/vla_wam_shared_v2/pilot/expansion/pi0_fast_current_stack_v2a008_release_probe.json",
  ],
];
probes.getRange("A7:J10").values = probeRows;
styleBody(probes.getRange("A7:J10"));
probes.tables.add("A6:J10", true, "InterfaceProbeTable");
probes.freezePanes.freezeRows(6);
probes.getRange("A:B").format.columnWidth = 30;
probes.getRange("C:C").format.columnWidth = 34;
probes.getRange("D:D").format.columnWidth = 18;
probes.getRange("E:F").format.columnWidth = 34;
probes.getRange("G:H").format.columnWidth = 20;
probes.getRange("I:I").format.columnWidth = 48;
probes.getRange("J:J").format.columnWidth = 72;
probes.getRange("A7:J10").format.wrapText = true;

// Definitions and source map -------------------------------------------------
setTitle(
  definitions,
  "A1:D2",
  "Definitions, claim boundaries, and sources",
  "The workbook is a reader-facing index of committed evidence, not a replacement for the frozen protocol or hash-bearing JSON.",
);
definitions.getRange("A6:D6").values = [["Item", "Definition / purpose", "Repository source", "GitHub URL"]];
styleHeader(definitions.getRange("A6:D6"));
const sourceRows = [
  [
    "Task success",
    "Arena-specific frozen completion predicate for the requested LEFT or RIGHT relation.",
    "artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison.json",
    "https://github.com/adeeb10abbas/steerable/blob/main/artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison.json",
  ],
  [
    "Endpoint alignment",
    "Within a matched state/seed pair, the RIGHT-command endpoint lies to the right of the LEFT-command endpoint. Sensitivity, not success.",
    "docs/VLA_WAM_STEERABILITY_V2_PROTOCOL.md",
    "https://github.com/adeeb10abbas/steerable/blob/main/docs/VLA_WAM_STEERABILITY_V2_PROTOCOL.md",
  ],
  [
    "Action distinctness",
    "The matched LEFT and RIGHT episodes executed different action traces. Missing/unreported is blank, never zero.",
    "docs/VLA_WAM_STEERABILITY_V2_PROTOCOL.md",
    "https://github.com/adeeb10abbas/steerable/blob/main/docs/VLA_WAM_STEERABILITY_V2_PROTOCOL.md",
  ],
  [
    "Behavioral denominator",
    "Valid completed episodes only. Setup failures, rendering failures, partial runs, and interface probes are excluded.",
    "artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison.md",
    "https://github.com/adeeb10abbas/steerable/blob/main/artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison.md",
  ],
  [
    "Arena separation",
    "DROID/RoboLab and RoboTwin use different tasks and success predicates; their raw success rates are never pooled.",
    "docs/VLA_WAM_RESEARCH_BLOG.md",
    "https://github.com/adeeb10abbas/steerable/blob/main/docs/VLA_WAM_RESEARCH_BLOG.md",
  ],
  [
    "Video roles",
    "Execution may evidence behavior. Prediction/imagination, alternate encodes, and reconstruction components are not additional episodes.",
    "artifacts/vla_wam_shared_v2/media/README.md",
    "https://github.com/adeeb10abbas/steerable/blob/main/artifacts/vla_wam_shared_v2/media/README.md",
  ],
  [
    "Cross-model source-set SHA-256",
    comparison.source_set_sha256,
    "artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison_manifest.json",
    "https://github.com/adeeb10abbas/steerable/blob/main/artifacts/vla_wam_shared_v2/results/direct_command_cross_model_comparison_manifest.json",
  ],
  [
    "Evidence cutoff commit",
    comparison.evidence_cutoff_git_commit,
    "Git branch main",
    "https://github.com/adeeb10abbas/steerable/tree/main",
  ],
];
definitions.getRange(`A7:D${6 + sourceRows.length}`).values = sourceRows;
styleBody(definitions.getRange(`A7:D${6 + sourceRows.length}`));
definitions.tables.add(`A6:D${6 + sourceRows.length}`, true, "DefinitionsSourcesTable");
definitions.freezePanes.freezeRows(6);
definitions.getRange("A:A").format.columnWidth = 28;
definitions.getRange("B:B").format.columnWidth = 72;
definitions.getRange("C:C").format.columnWidth = 76;
definitions.getRange("D:D").format.columnWidth = 88;
definitions.getRange(`A7:D${6 + sourceRows.length}`).format.wrapText = true;

// Dashboard -----------------------------------------------------------------
setTitle(
  dashboard,
  "A1:N2",
  "VLA/WAM language-steerability evidence dashboard",
  "Success, endpoint alignment, action distinctness, and video availability. All rate calculations link to the behavioral source sheet; arenas remain separate.",
);
dashboard.getRange("A6:C6").values = [["Behavioral models", "DROID valid episodes", "RoboTwin valid episodes"]];
styleHeader(dashboard.getRange("A6:C6"));
dashboard.getRange("A7:C7").formulas = [[
  `=COUNTA('Behavioral Results'!C7:C${6 + resultRows.length})`,
  `=SUMIF('Behavioral Results'!A7:A${6 + resultRows.length},"DROID / RoboLab",'Behavioral Results'!D7:D${6 + resultRows.length})`,
  `=SUMIF('Behavioral Results'!A7:A${6 + resultRows.length},"RoboTwin place-A-relative-to-B",'Behavioral Results'!D7:D${6 + resultRows.length})`,
]];
dashboard.getRange("A7:C7").format = {
  fill: COLORS.paleTeal,
  font: { bold: true, color: COLORS.navy, size: 18 },
  horizontalAlignment: "center",
  borders: { preset: "all", style: "thin", color: COLORS.border },
};
dashboard.getRange("E6:G6").values = [["Canonical execution videos", "Canonical predictions", "Committed MP4s"]];
styleHeader(dashboard.getRange("E6:G6"));
dashboard.getRange("E7:G7").formulas = [[
  `=COUNTIFS('Media Inventory'!D7:D${6 + mediaRows.length},"actual rollout",'Media Inventory'!F7:F${6 + mediaRows.length},"Yes")`,
  `=COUNTIFS('Media Inventory'!F7:F${6 + mediaRows.length},"Yes")-E7`,
  `=COUNTA('Media Inventory'!G7:G${6 + mediaRows.length})`,
]];
dashboard.getRange("E7:G7").format = {
  fill: COLORS.paleOrange,
  font: { bold: true, color: COLORS.navy, size: 18 },
  horizontalAlignment: "center",
  borders: { preset: "all", style: "thin", color: COLORS.border },
};

const droidRows = comparison.rows
  .map((row, index) => ({ row, sheetRow: 7 + index }))
  .filter(({ row }) => row.arena_id === "droid_robolab");
const robotwinRows = comparison.rows
  .map((row, index) => ({ row, sheetRow: 7 + index }))
  .filter(({ row }) => row.arena_id === "robotwin_place_a2b");

dashboard.getRange("A11:C11").values = [["DROID / RoboLab model", "Overall success", "Endpoint alignment"]];
styleHeader(dashboard.getRange("A11:C11"));
droidRows.forEach(({ sheetRow }, index) => {
  const row = 12 + index;
  dashboard.getRange(`A${row}:C${row}`).formulas = [[
    `='Behavioral Results'!C${sheetRow}`,
    `='Behavioral Results'!M${sheetRow}`,
    `='Behavioral Results'!P${sheetRow}`,
  ]];
});
dashboard.getRange(`B12:C${11 + droidRows.length}`).format.numberFormat = "0%";
styleBody(dashboard.getRange(`A12:C${11 + droidRows.length}`));
const droidChart = dashboard.charts.add("bar", dashboard.getRange(`A11:B${11 + droidRows.length}`));
droidChart.title = "DROID success by checkpoint";
droidChart.hasLegend = false;
droidChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
droidChart.yAxis = { numberFormatCode: "0%", min: 0, max: 1 };
droidChart.setPosition("E10", "N25");

dashboard.getRange("A28:C28").values = [["RoboTwin model", "Overall success", "Endpoint alignment"]];
styleHeader(dashboard.getRange("A28:C28"));
robotwinRows.forEach(({ sheetRow }, index) => {
  const row = 29 + index;
  dashboard.getRange(`A${row}:C${row}`).formulas = [[
    `='Behavioral Results'!C${sheetRow}`,
    `='Behavioral Results'!M${sheetRow}`,
    `='Behavioral Results'!P${sheetRow}`,
  ]];
});
dashboard.getRange(`B29:C${28 + robotwinRows.length}`).format.numberFormat = "0%";
styleBody(dashboard.getRange(`A29:C${28 + robotwinRows.length}`));
const robotwinChart = dashboard.charts.add("bar", dashboard.getRange(`A28:B${28 + robotwinRows.length}`));
robotwinChart.title = "RoboTwin success by checkpoint";
robotwinChart.hasLegend = false;
robotwinChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
robotwinChart.yAxis = { numberFormatCode: "0%", min: 0, max: 1 };
robotwinChart.setPosition("E27", "N42");

dashboard.getRange("A45:N47").merge();
dashboard.getRange("A45").values = [[
  "Interpretation: many checkpoints changed actions and correctly ordered endpoints even when task completion failed. Prediction/imagination media is therefore shown beside—but never counted as—execution.",
]];
dashboard.getRange("A45:N47").format = {
  fill: COLORS.paleGray,
  font: { color: COLORS.text, italic: true, size: 11 },
  wrapText: true,
  verticalAlignment: "center",
  borders: { preset: "outside", style: "thin", color: COLORS.border },
};
dashboard.getRange("A:A").format.columnWidth = 42;
dashboard.getRange("B:C").format.columnWidth = 18;
dashboard.getRange("D:D").format.columnWidth = 3;
dashboard.getRange("E:N").format.columnWidth = 13;
dashboard.freezePanes.freezeRows(5);

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

for (const sheetName of [
  "Dashboard",
  "Behavioral Results",
  "Media Inventory",
  "Interface Probes",
  "Definitions & Sources",
]) {
  const preview = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: sheetName === "Dashboard" ? 1 : 0.75,
    format: "png",
  });
  const safeName = sheetName.toLowerCase().replaceAll(/[^a-z0-9]+/g, "_");
  await fs.writeFile(
    path.join(previewDir, `${safeName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const formulaInspection = await workbook.inspect({
  kind: "formula",
  sheetId: "Behavioral Results",
  range: `A1:X${6 + resultRows.length}`,
  maxChars: 6000,
  options: { maxResults: 200 },
});
console.log(formulaInspection.ndjson);
const dashboardInspection = await workbook.inspect({
  kind: "region",
  sheetId: "Dashboard",
  range: "A1:N47",
  maxChars: 6000,
});
console.log(dashboardInspection.ndjson);

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
const outputPath = path.join(outputDir, "vla_wam_study_stats.xlsx");
await xlsx.save(outputPath);
console.log(JSON.stringify({
  outputPath,
  previewDir,
  behavioralRows: resultRows.length,
  mediaRows: mediaRows.length,
  predictionOnlyContracts: gallery.prediction_only_manifest_contracts.length,
}));
