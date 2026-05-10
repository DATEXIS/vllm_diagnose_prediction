import { NextResponse } from "next/server";
import { loadIcdBundleDisk } from "@/lib/icd";

export function GET() {
  const bundle = loadIcdBundleDisk();
  const [exact10, cat10, exact9, cat9] = bundle;
  return NextResponse.json({ exact10, cat10, exact9, cat9 });
}
