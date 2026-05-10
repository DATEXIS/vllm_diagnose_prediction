import path from "node:path";
import { NextResponse } from "next/server";
import { getUiRoot } from "@/lib/env";
import { parseRunPath, runIdFromPath } from "@/lib/runPath";
import { fetchRunFromWandb } from "@/lib/pythonFetch";
import { readSampleRecords } from "@/lib/sampleTable";

export async function POST(req: Request) {
  const body = (await req.json()) as { runInput?: string };
  const runInput = body.runInput;
  if (!runInput || typeof runInput !== "string")
    return NextResponse.json({ error: "runInput required" }, { status: 400 });
  const runPath = parseRunPath(runInput);
  fetchRunFromWandb(runInput);
  const runId = runIdFromPath(runPath);
  const runDir = path.join(getUiRoot(), "wandb_cache", runId);
  const { records } = readSampleRecords(runDir);
  return NextResponse.json({
    runId,
    runPath,
    patientCount: records.length,
  });
}
