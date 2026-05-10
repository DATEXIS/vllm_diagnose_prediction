import path from "node:path";
import fs from "node:fs";
import { parse } from "csv-parse/sync";
import { NextResponse } from "next/server";
import { findFileRecursive } from "@/lib/fsFind";
import { getUiRoot } from "@/lib/env";

const CHART_COLS = [
  "iter/all/f1_micro",
  "iter/all/f1_macro",
  "retrieval_pct/semantic",
  "retrieval_pct/threshold_fpr",
  "retrieval_pct/threshold_fnr",
  "iter/all/parse_failures",
];

export async function GET(
  _req: Request,
  { params }: { params: { runId: string } },
) {
  const runDir = path.join(getUiRoot(), "wandb_cache", params.runId);

  const summaryPath = findFileRecursive(
    runDir,
    (name) => name === "wandb-summary.json",
  );
  let summary: unknown = {};
  if (summaryPath) summary = JSON.parse(fs.readFileSync(summaryPath, "utf8"));

  const logHit = findFileRecursive(runDir, (name) => name === "output.log");
  let logTail = "";
  if (logHit) {
    const lines = fs.readFileSync(logHit, "utf8").split("\n");
    logTail = lines.slice(-200).join("\n");
  }

  const histPath = path.join(runDir, "history_metrics.csv");
  let chart: { cols: string[]; rows: Record<string, unknown>[] } | null = null;
  if (fs.existsSync(histPath)) {
    const raw = fs.readFileSync(histPath, "utf8");
    const rows = parse(raw, {
      columns: true,
      skip_empty_lines: true,
      relax_column_count: true,
    }) as Record<string, unknown>[];
    const cols = CHART_COLS.filter((c) => rows.length > 0 && c in rows[0]!);
    chart = cols.length ? { cols, rows } : { cols: [], rows };
  }

  return NextResponse.json({
    summary,
    logTail,
    chart,
  });
}
