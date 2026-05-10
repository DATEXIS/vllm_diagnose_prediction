import path from "node:path";
import fs from "node:fs";
import { NextResponse } from "next/server";
import { getUiRoot } from "@/lib/env";
import {
  DEFAULT_RUN_ID,
  defaultRunUrl,
  parseRunPath,
  runIdFromPath,
} from "@/lib/runPath";
import { fetchRunFromWandb } from "@/lib/pythonFetch";
import { readSampleRecords } from "@/lib/sampleTable";

function defaultRunReady(uiRoot: string): boolean {
  const dir = path.join(uiRoot, "wandb_cache", DEFAULT_RUN_ID);
  return fs.existsSync(path.join(dir, "wandb-summary.json"));
}

export async function GET() {
  const uiRoot = getUiRoot();
  if (!defaultRunReady(uiRoot)) fetchRunFromWandb(defaultRunUrl());
  const runPath = parseRunPath(DEFAULT_RUN_ID);
  const runId = runIdFromPath(runPath);
  const runDir = path.join(uiRoot, "wandb_cache", runId);
  const { records } = readSampleRecords(runDir);
  return NextResponse.json({
    runId,
    runPath,
    patientCount: records.length,
    defaultRunInput: defaultRunUrl(),
  });
}
