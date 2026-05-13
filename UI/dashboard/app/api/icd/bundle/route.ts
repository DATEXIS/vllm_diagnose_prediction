import { NextResponse } from "next/server";
import { loadIcdBundleDisk } from "@/lib/icd";

/** Must run at runtime: builder has no `UI/ICD_names`; disk path uses `MERLIN_UI_ROOT` in the image. */
export const dynamic = "force-dynamic";

export function GET() {
  const bundle = loadIcdBundleDisk();
  const [exact10, cat10, exact9, cat9] = bundle;
  return NextResponse.json({ exact10, cat10, exact9, cat9 });
}
