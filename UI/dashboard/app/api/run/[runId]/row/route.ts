import path from "node:path";
import { NextResponse } from "next/server";
import { getUiRoot } from "@/lib/env";
import { readSampleRecords } from "@/lib/sampleTable";

export async function GET(
  req: Request,
  { params }: { params: { runId: string } },
) {
  const { searchParams } = new URL(req.url);
  const index = parseInt(searchParams.get("index") ?? "1", 10);
  if (Number.isNaN(index) || index < 1)
    return NextResponse.json({ error: "bad index" }, { status: 400 });

  const runDir = path.join(getUiRoot(), "wandb_cache", params.runId);
  const { records } = readSampleRecords(runDir);
  if (index > records.length)
    return NextResponse.json({ error: "index out of range" }, { status: 400 });
  return NextResponse.json({ row: records[index - 1]! });
}
