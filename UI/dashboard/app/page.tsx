"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleCheck,
  CircleHelp,
  CircleX,
  ClipboardList,
  FileText,
  Loader2,
  Settings2,
  ShieldAlert,
} from "lucide-react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  Legend,
  XAxis,
  YAxis,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { IcdMaps } from "@/lib/icdResolve";
import { formatCodeLine } from "@/lib/icdResolve";
import {
  buildIterationRecords,
  coerceGroundTruthCodes,
  extractNoteSummary,
  normalizeCode,
  type IterationRecord,
} from "@/lib/replay";
import { defaultRunUrl } from "@/lib/runPath";

const SCALE_HEIGHTS = {
  Small: { note: 280, list: 200 },
  Medium: { note: 360, list: 260 },
  Large: { note: 460, list: 340 },
} as const;

type PaneScale = keyof typeof SCALE_HEIGHTS | "Custom";

function sliderFirst(v: number | readonly number[]): number {
  return Array.isArray(v) ? Number(v[0]) : Number(v);
}

function scrollBoxStyle(h: number, extra = 100) {
  return {
    minHeight: h,
    maxHeight: `min(38vh, ${h + extra}px)`,
  } as const;
}

function metricsFromCodes(pred: string[], gt: string[]) {
  const pset = new Set(pred);
  const gset = new Set(gt);
  let tp = 0;
  for (const c of pset) {
    if (gset.has(c)) tp += 1;
  }
  const fp = pset.size - tp;
  let fn = 0;
  for (const c of gset) {
    if (!pset.has(c)) fn += 1;
  }
  const prec = tp + fp === 0 ? null : tp / (tp + fp);
  const rec = tp + fn === 0 ? null : tp / (tp + fn);
  const f1 =
    prec != null && rec != null && prec + rec > 0
      ? (2 * prec * rec) / (prec + rec)
      : null;
  return { tp, fp, fn, prec, rec, f1 };
}

function formatScore(n: number | null, digits = 3) {
  return n == null || !Number.isFinite(n) ? "—" : n.toFixed(digits);
}

function patientTitleFromRow(row: Record<string, unknown> | null, index: number) {
  if (!row) return `Patient ${String(index).padStart(5, "0")}`;
  const note = String(row.admission_note ?? "").trim();
  if (!note) return `Patient ${String(index).padStart(5, "0")}`;
  const first =
    note
      .split("\n")
      .map((ln) => ln.trim())
      .filter(Boolean)[0] ?? "";
  const cleaned = first.replace(/^CHIEF COMPLAINT:\s*/i, "").trim();
  const short =
    cleaned.length > 90 ? `${cleaned.slice(0, 90).trim()}…` : cleaned;
  return `Patient ${String(index).padStart(5, "0")}, ${short}`;
}

/** Split ICD label line into box chip + prose (matches reference two-column decks). */
function splitCodeParts(maps: IcdMaps, code: string) {
  const line = formatCodeLine(maps, code);
  const sep = " — ";
  const idx = line.indexOf(sep);
  if (idx >= 0) {
    return { chip: line.slice(0, idx).trim(), desc: line.slice(idx + sep.length).trim() };
  }
  return { chip: code.trim(), desc: "" };
}

const predictionPanelStyles = {
  tp: {
    card: "border-emerald-200/55 bg-emerald-50/[0.97] shadow-sm shadow-emerald-900/[0.04]",
    badge:
      "border border-emerald-300/35 bg-emerald-200/50 text-emerald-950 hover:bg-emerald-200/50",
    Icon: CircleCheck,
    iconWrap: "bg-emerald-100 text-emerald-700 ring-1 ring-emerald-200/60",
    chip:
      "border border-emerald-200/55 bg-white text-emerald-950 shadow-sm shadow-emerald-900/10",
  },
  fp: {
    card: "border-rose-200/55 bg-rose-50/[0.97] shadow-sm shadow-rose-900/[0.04]",
    badge:
      "border border-rose-300/35 bg-rose-200/50 text-rose-950 hover:bg-rose-200/50",
    Icon: CircleX,
    iconWrap: "bg-rose-100 text-rose-700 ring-1 ring-rose-200/60",
    chip:
      "border border-rose-200/55 bg-white text-rose-950 shadow-sm shadow-rose-900/10",
  },
  fn: {
    card: "border-amber-200/55 bg-amber-50/[0.97] shadow-sm shadow-amber-900/[0.04]",
    badge:
      "border border-amber-300/40 bg-amber-200/45 text-amber-950 hover:bg-amber-200/45",
    Icon: AlertTriangle,
    iconWrap: "bg-amber-100 text-amber-800 ring-1 ring-amber-200/60",
    chip:
      "border border-amber-200/55 bg-white text-amber-950 shadow-sm shadow-amber-900/10",
  },
  gt: {
    card: "border-sky-200/50 bg-sky-50/[0.95] shadow-sm shadow-sky-900/[0.04]",
    badge:
      "border border-sky-300/35 bg-sky-200/45 text-sky-950 hover:bg-sky-200/45",
    Icon: ClipboardList,
    iconWrap: "bg-sky-100 text-sky-800 ring-1 ring-sky-200/60",
    chip:
      "border border-sky-200/55 bg-white text-sky-950 shadow-sm shadow-sky-900/10",
  },
} as const;

function PredictionPanelCard({
  kind,
  title,
  count,
  codes,
  maps,
  empty,
  scrollMinH,
}: {
  kind: keyof typeof predictionPanelStyles;
  title: string;
  count: number;
  codes: string[];
  maps: IcdMaps;
  empty: string;
  scrollMinH: number;
}) {
  const s = predictionPanelStyles[kind];
  const PanelIcon = s.Icon;
  return (
    <div
      className={cn(
        "animate-in fade-in-0 slide-in-from-bottom-1 rounded-2xl border duration-300",
        "transition-[box-shadow,transform] hover:-translate-y-px hover:shadow-md",
        s.card,
      )}
    >
      <div className="flex items-center gap-3 border-b border-black/[0.04] px-4 py-3.5 sm:px-5">
        <div
          className={cn(
            "flex size-10 shrink-0 items-center justify-center rounded-xl [&_svg]:size-5",
            s.iconWrap,
          )}
        >
          <PanelIcon strokeWidth={2.25} aria-hidden />
        </div>
        <h3 className="min-w-0 flex-1 text-base font-semibold tracking-tight text-foreground">
          {title}
        </h3>
        <Badge
          variant="secondary"
          className={cn(
            "shrink-0 rounded-full px-2.5 py-1 tabular-nums text-xs font-bold",
            s.badge,
          )}
        >
          {count}
        </Badge>
      </div>
      <div className="px-4 pb-4 pt-3 sm:px-5">
        <div
          className="overflow-y-auto pr-1 [scrollbar-gutter:stable]"
          style={scrollBoxStyle(scrollMinH, 48)}
        >
          {codes.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted-foreground">{empty}</p>
          ) : (
            <ul className="space-y-3">
              {codes.map((c) => {
                const { chip, desc } = splitCodeParts(maps, c);
                return (
                  <li key={c} className="flex gap-3 text-sm leading-snug">
                    <span
                      className={cn(
                        "flex min-h-[2rem] min-w-[2.75rem] shrink-0 items-center justify-center rounded-lg px-2 py-1 text-center text-xs font-bold tabular-nums tracking-tight",
                        s.chip,
                      )}
                    >
                      {chip}
                    </span>
                    <span className="pt-1 text-[13px] text-foreground/90">
                      {desc || "(no description loaded)"}
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

function RefMetric({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div
      className={cn(
        "animate-in fade-in-0 zoom-in-95 rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm duration-300",
        "transition-transform hover:-translate-y-0.5 hover:shadow-md",
      )}
    >
      <Tooltip>
        <TooltipTrigger className="flex w-full cursor-help items-center gap-1.5 text-left">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            {label}
          </span>
          <CircleHelp className="size-3.5 text-muted-foreground/80" />
        </TooltipTrigger>
        <TooltipContent side="bottom" className="max-w-xs text-xs leading-relaxed">
          {hint}
        </TooltipContent>
      </Tooltip>
      <p className="mt-3 text-3xl font-semibold tabular-nums tracking-tight text-foreground">
        {value}
      </p>
    </div>
  );
}

export default function HomePage() {
  const [maps, setMaps] = useState<IcdMaps | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [patientCount, setPatientCount] = useState(0);
  const [runInput, setRunInput] = useState(defaultRunUrl());
  const [row, setRow] = useState<Record<string, unknown> | null>(null);
  const [bootErr, setBootErr] = useState<string | null>(null);
  const [loadMsg, setLoadMsg] = useState<string | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [loadPending, setLoadPending] = useState(false);
  const [patientIndex, setPatientIndex] = useState(1);
  const [paneScale, setPaneScale] = useState<PaneScale>("Medium");
  const [customHeight, setCustomHeight] = useState(380);
  const [currentT, setCurrentT] = useState(0);
  const [showFullNote, setShowFullNote] = useState(false);
  const [diagOpen, setDiagOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [instrTab, setInstrTab] = useState<"new" | "accumulated">("new");
  const [diag, setDiag] = useState<{
    summary: unknown;
    logTail: string;
    chart: { cols: string[]; rows: Record<string, unknown>[] } | null;
  } | null>(null);

  useEffect(() => {
    void (async () => {
      const r = await fetch("/api/icd/bundle");
      setMaps((await r.json()) as IcdMaps);
    })();
  }, []);

  useEffect(() => {
    const cap = Math.max(1, patientCount || 1);
    setPatientIndex((i) =>
      Number.isFinite(i) ? Math.min(Math.max(1, i), cap) : 1,
    );
  }, [patientCount]);

  useEffect(() => {
    void (async () => {
      setBootErr(null);
      const r = await fetch("/api/run/bootstrap");
      if (!r.ok) {
        setBootErr(await r.text());
        return;
      }
      const j = (await r.json()) as {
        runId: string;
        patientCount: number;
        defaultRunInput: string;
      };
      setRunId(j.runId);
      setPatientCount(j.patientCount);
      setRunInput(j.defaultRunInput);
      setPatientIndex(1);
    })();
  }, []);

  useEffect(() => {
    if (!runId) return;
    void (async () => {
      const r = await fetch(
        `/api/run/${encodeURIComponent(runId)}/row?index=${patientIndex}`,
      );
      const j = (await r.json()) as { row?: Record<string, unknown> };
      setRow(r.ok && j.row ? j.row : null);
    })();
  }, [runId, patientIndex]);

  useEffect(() => {
    if (!diagOpen || !runId) return;
    void (async () => {
      const r = await fetch(`/api/run/${encodeURIComponent(runId)}/diagnostics`);
      setDiag(await r.json());
    })();
  }, [diagOpen, runId]);

  const iterations = buildIterationRecords(row ?? {});
  const maxT =
    iterations.length === 0
      ? 0
      : Math.max(...iterations.map((r) => r.iteration));

  useEffect(() => {
    setCurrentT(maxT);
  }, [maxT, patientIndex, runId, row]);

  const byT = useMemo(() => {
    const m = new Map<number, IterationRecord>();
    for (const r of iterations) m.set(r.iteration, r);
    return m;
  }, [iterations]);

  const stepKeys = useMemo(() => [...byT.keys()].sort((a, b) => a - b), [byT]);

  const selected =
    iterations.length === 0
      ? null
      : (byT.get(currentT) ?? iterations[iterations.length - 1]!);

  const heights = useMemo(() => {
    if (paneScale === "Custom") {
      const h = Math.max(140, customHeight);
      return { note: h, list: Math.floor(h * 0.72) };
    }
    return SCALE_HEIGHTS[paneScale as keyof typeof SCALE_HEIGHTS];
  }, [customHeight, paneScale]);

  const gtCodesSorted = useMemo(() => {
    if (!row) return [];
    const gtCell = row.true_codes ?? row.ICD_CODES;
    return [
      ...new Set(
        coerceGroundTruthCodes(gtCell).map(normalizeCode).filter(Boolean),
      ),
    ].sort();
  }, [row]);

  const predCodesSorted = selected
    ? [...new Set(selected.prediction.map(normalizeCode).filter(Boolean))].sort()
    : [];

  const tpCodes = predCodesSorted.filter((c) => gtCodesSorted.includes(c));
  const fpCodes = predCodesSorted.filter((c) => !gtCodesSorted.includes(c));
  const fnCodes = gtCodesSorted.filter((c) => !predCodesSorted.includes(c));

  const cumulativeInstructions = useMemo(() => {
    let acc: string[] = [];
    for (let t = 0; t <= currentT; t++) {
      const rec = byT.get(t);
      if (rec) acc = [...rec.instructions, ...acc];
    }
    return acc;
  }, [byT, currentT]);

  const newInstructions = selected?.instructions ?? [];

  const { prec, rec, f1 } = metricsFromCodes(predCodesSorted, gtCodesSorted);

  const patientTitle = patientTitleFromRow(row, patientIndex);

  const f1ByStep = useMemo(() => {
    const map = new Map<number, number | null>();
    for (const t of stepKeys) {
      const rec = byT.get(t);
      const preds = rec
        ? [...new Set(rec.prediction.map(normalizeCode).filter(Boolean))].sort()
        : [];
      const { f1: fv } = metricsFromCodes(preds, gtCodesSorted);
      map.set(t, fv);
    }
    return map;
  }, [byT, gtCodesSorted, stepKeys]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (
        document.activeElement instanceof HTMLInputElement ||
        document.activeElement instanceof HTMLTextAreaElement ||
        document.activeElement instanceof HTMLSelectElement
      ) {
        return;
      }
      if (!row || stepKeys.length === 0) return;
      const minT = stepKeys[0];
      const maxStep = stepKeys[stepKeys.length - 1]!;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        setCurrentT((prev) => {
          const ix = stepKeys.indexOf(prev);
          if (ix >= 0 && ix < stepKeys.length - 1) return stepKeys[ix + 1]!;
          return maxStep;
        });
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setCurrentT((prev) => {
          const ix = stepKeys.indexOf(prev);
          if (ix > 0) return stepKeys[ix - 1]!;
          return minT ?? prev;
        });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [row, stepKeys]);

  const onLoadRun = useCallback(async () => {
    setLoadErr(null);
    setLoadMsg(null);
    setLoadPending(true);
    const r = await fetch("/api/run/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ runInput }),
    });
    setLoadPending(false);
    if (!r.ok) {
      setLoadErr(await r.text());
      return;
    }
    const j = (await r.json()) as {
      runId: string;
      patientCount: number;
      runPath: string;
    };
    setRunId(j.runId);
    setPatientCount(j.patientCount);
    setPatientIndex(1);
    setLoadMsg(`Loaded ${j.runPath}`);
  }, [runInput]);

  const chartData = useMemo(() => {
    if (!diag?.chart?.cols?.length) return [];
    return diag.chart.rows.map((rowRow, i) => {
      const pt: Record<string, number | string> = { step: i };
      for (const c of diag.chart!.cols) {
        const v = rowRow[c];
        const n = typeof v === "number" ? v : parseFloat(String(v));
        pt[c] = Number.isFinite(n) ? n : 0;
      }
      return pt;
    });
  }, [diag]);

  const defaultMaps: IcdMaps = {
    exact10: {},
    cat10: {},
    exact9: {},
    cat9: {},
  };
  const m = maps ?? defaultMaps;

  const safePatientCap = Math.max(1, patientCount || 1);
  const noteText = row
    ? showFullNote
      ? String(row.admission_note ?? "")
      : extractNoteSummary(String(row.admission_note ?? ""), 10)
    : "";

  const canPrevPatient = patientIndex > 1;
  const canNextPatient = patientIndex < safePatientCap;

  const tabBtn = (
    id: typeof instrTab,
    label: string,
    badge?: number,
  ) => (
    <button
      type="button"
      key={id}
      onClick={() => setInstrTab(id)}
      className={cn(
        "relative flex flex-1 items-center justify-center gap-2 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-200 ease-out",
        instrTab === id
          ? "bg-white text-foreground shadow-sm ring-1 ring-slate-200/80"
          : "text-muted-foreground hover:text-foreground",
      )}
    >
      {label}
      {badge != null ? (
        <span className="tabular-nums text-xs font-semibold text-muted-foreground">
          {badge}
        </span>
      ) : null}
    </button>
  );

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100/80">
      <div className="mx-auto max-w-6xl px-4 pb-16 pt-8 sm:px-6 lg:px-8">
        <header className="animate-in fade-in-0 slide-in-from-top-2 duration-500">
          <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
            <div className="flex gap-3.5">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-sky-100 shadow-sm ring-1 ring-sky-200/60 transition-transform duration-300 hover:scale-105">
                <Activity className="size-6 text-sky-600" strokeWidth={2.2} />
              </div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-foreground sm:text-3xl">
                  Diagnosis Inference Dashboard
                </h1>
                <p className="mt-1 text-sm text-muted-foreground">
                  Iterative ICD prediction · agent run viewer
                </p>
              </div>
            </div>

            <div className="flex flex-col items-stretch gap-3 sm:items-end">
              <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                Currently viewing
              </p>
              <h2 className="max-w-md text-left text-base font-semibold leading-snug text-foreground sm:max-w-sm sm:text-right">
                {patientTitle}
              </h2>
              <div className="flex items-center justify-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-9 gap-1 rounded-xl border-slate-200 bg-white px-3 shadow-sm transition-all duration-200 hover:bg-slate-50"
                  disabled={!canPrevPatient}
                  onClick={() =>
                    setPatientIndex((i) => Math.max(1, i - 1))
                  }
                >
                  <ChevronLeft className="size-4" />
                  Prev
                </Button>
                <span className="min-w-[4.5rem] text-center text-sm font-medium tabular-nums text-foreground">
                  {patientIndex} / {safePatientCap}
                </span>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-9 gap-1 rounded-xl border-slate-200 bg-white px-3 shadow-sm transition-all duration-200 hover:bg-slate-50"
                  disabled={!canNextPatient}
                  onClick={() =>
                    setPatientIndex((i) => Math.min(safePatientCap, i + 1))
                  }
                >
                  Next
                  <ChevronRight className="size-4" />
                </Button>
              </div>
            </div>
          </div>

          <div className="mt-5 flex justify-end">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="gap-2 rounded-xl text-muted-foreground hover:text-foreground"
              onClick={() => setSettingsOpen((o) => !o)}
            >
              <Settings2 className="size-4" />
              Run &amp; layout
              <ChevronDown
                className={cn(
                  "size-4 transition-transform duration-200",
                  settingsOpen && "rotate-180",
                )}
              />
            </Button>
          </div>

          <Collapsible open={settingsOpen} onOpenChange={setSettingsOpen}>
            <CollapsibleContent className="data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:slide-in-from-top-1">
              <Card className="mt-4 border-slate-200/90 shadow-md ring-0">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Load W&amp;B run</CardTitle>
                  <CardDescription className="text-xs">
                    Entity / project / run id or filesystem path handled by the
                    server.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="run-url">Run URL or path</Label>
                    <Input
                      id="run-url"
                      value={runInput}
                      onChange={(e) => setRunInput(e.target.value)}
                      className="font-mono text-xs"
                      placeholder="entity/project/run_id"
                    />
                  </div>
                  <Button
                    type="button"
                    className="w-full sm:w-auto"
                    disabled={loadPending}
                    onClick={() => void onLoadRun()}
                  >
                    {loadPending ? (
                      <>
                        <Loader2 className="mr-2 size-4 animate-spin" />
                        Loading…
                      </>
                    ) : (
                      "Load run"
                    )}
                  </Button>
                  {loadMsg ? (
                    <p className="text-xs text-emerald-700 dark:text-emerald-400">
                      {loadMsg}
                    </p>
                  ) : null}
                  {loadErr ? (
                    <p className="whitespace-pre-wrap text-xs text-destructive">
                      {loadErr}
                    </p>
                  ) : null}
                  {bootErr ? (
                    <p className="whitespace-pre-wrap text-xs text-destructive">
                      Bootstrap: {bootErr}
                    </p>
                  ) : null}
                  <Separator />
                  {/* <div className="grid gap-4 sm:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Pane size</Label>
                      <Select
                        value={paneScale}
                        onValueChange={(v) => setPaneScale(v as PaneScale)}
                      >
                        <SelectTrigger className="w-full rounded-xl">
                          <SelectValue placeholder="Size" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="Small">Small</SelectItem>
                          <SelectItem value="Medium">Medium</SelectItem>
                          <SelectItem value="Large">Large</SelectItem>
                          <SelectItem value="Custom">Custom</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    {paneScale === "Custom" ? (
                      <div className="space-y-3">
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <Label>Base scroll height</Label>
                          <span>{customHeight}px</span>
                        </div>
                        <Slider
                          value={[customHeight]}
                          min={180}
                          max={700}
                          step={20}
                          onValueChange={(v) =>
                            setCustomHeight(
                              Number.isFinite(sliderFirst(v))
                                ? sliderFirst(v)
                                : 380,
                            )
                          }
                        />
                      </div>
                    ) : null}
                  </div> */}
                  <p className="text-[11px] text-muted-foreground">
                    Active run{" "}
                    <code className="rounded-md bg-muted px-1.5 py-0.5 font-mono text-[10px]">
                      {runId ?? "—"}
                    </code>
                  </p>
                </CardContent>
              </Card>
            </CollapsibleContent>
          </Collapsible>
        </header>

        <section className="mt-10 space-y-8">
          {row ? (
            <div
              className={cn(
                "flex gap-3 rounded-2xl border border-amber-200/70 bg-amber-50/90 px-4 py-3.5 shadow-sm",
                "animate-in fade-in-0 slide-in-from-top-1 duration-300",
              )}
            >
              <ShieldAlert
                className="mt-0.5 size-5 shrink-0 text-amber-700"
                strokeWidth={2}
                aria-hidden
              />
              <div className="min-w-0">
                <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-amber-900/75">
                  Verifier stop reason
                </p>
                <p className="mt-1 break-words text-sm font-medium leading-snug text-foreground">
                  {String(row.halt_reason ?? "—")}
                </p>
              </div>
            </div>
          ) : null}

          {!row || iterations.length === 0 ? (
            <Card className="border-dashed border-slate-300 shadow-sm">
              <CardContent className="py-16 text-center">
                <p className="text-muted-foreground">
                  No iteration records for this patient yet, or still loading.
                </p>
                <p className="mt-2 text-sm text-muted-foreground">
                  Open <span className="font-medium">Run &amp; layout</span>{" "}
                  above to load a different W&amp;B run.
                </p>
              </CardContent>
            </Card>
          ) : (
            <>
              <div
                className={cn(
                  "animate-in fade-in-0 slide-in-from-bottom-2 rounded-2xl border border-slate-200/90 bg-white p-6 shadow-sm duration-500",
                )}
              >
                <div className="relative flex justify-between gap-2 pt-2">
                  <div
                    className="pointer-events-none absolute left-[8%] right-[8%] top-5 h-px bg-slate-200"
                    aria-hidden
                  />
                  {stepKeys.map((t) => {
                    const active = currentT === t;
                    const f1s = f1ByStep.get(t);
                    const stepNo = stepKeys.indexOf(t) + 1;
                    return (
                      <div
                        key={t}
                        className="relative z-10 flex min-w-0 flex-1 flex-col items-center"
                      >
                        <button
                          type="button"
                          onClick={() => setCurrentT(t)}
                          className={cn(
                            "flex h-10 w-10 shrink-0 items-center justify-center rounded-full border-2 text-sm font-bold transition-all duration-200 ease-out",
                            active
                              ? "scale-110 border-primary bg-primary text-primary-foreground shadow-md ring-4 ring-primary/15"
                              : "border-slate-200 bg-white text-slate-500 hover:scale-105 hover:border-primary/35 hover:text-foreground",
                          )}
                        >
                          {stepNo}
                        </button>
                        <p
                          className={cn(
                            "mt-3 text-center text-xs font-semibold transition-colors duration-200",
                            active ? "text-foreground" : "text-muted-foreground",
                          )}
                        >
                          Iteration {stepNo}
                          <span className="block font-normal text-[10px] text-muted-foreground/90">
                            t={t}
                          </span>
                        </p>
                        <p
                          className={cn(
                            "mt-0.5 text-center text-[11px] tabular-nums transition-colors duration-200",
                            active ? "text-foreground" : "text-muted-foreground",
                          )}
                        >
                          F1 {formatScore(f1s ?? null)}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-3">
                <RefMetric
                  label="Precision"
                  value={formatScore(prec)}
                  hint="TP / (TP + FP) at the selected iteration using 3-char ICD buckets."
                />
                <RefMetric
                  label="Recall"
                  value={formatScore(rec)}
                  hint="TP / (TP + FN) at the selected iteration."
                />
                <RefMetric
                  label="F1 score"
                  value={formatScore(f1)}
                  hint="Harmonic mean of precision and recall for this timestep."
                />
              </div>

              <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-12 lg:gap-8">
                <div
                  className={cn(
                    "animate-in fade-in-0 slide-in-from-bottom-2 lg:col-span-7 xl:col-span-7",
                    "rounded-2xl border border-slate-200/90 bg-white shadow-sm duration-500",
                    "transition-shadow hover:shadow-md",
                  )}
                >
                  <div className="flex flex-wrap items-center gap-3 border-b border-slate-100 px-5 py-4">
                    <div className="flex size-11 shrink-0 items-center justify-center rounded-xl bg-sky-100 ring-1 ring-sky-200/60">
                      <FileText
                        className="size-[1.35rem] text-sky-600"
                        strokeWidth={2}
                        aria-hidden
                      />
                    </div>
                    <h3 className="flex-1 text-lg font-semibold tracking-tight">
                      Admission note
                    </h3>
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="full-note"
                        checked={showFullNote}
                        onCheckedChange={(c) => setShowFullNote(c === true)}
                      />
                      <Label htmlFor="full-note" className="text-sm font-normal">
                        Full note
                      </Label>
                    </div>
                  </div>
                  <div className="px-5 py-5">
                    <div
                      className="overflow-y-auto rounded-xl border border-slate-100 bg-white [scrollbar-gutter:stable]"
                      style={scrollBoxStyle(heights.note, 80)}
                    >
                      <p className="whitespace-pre-wrap px-4 py-4 font-mono text-[13px] leading-relaxed text-foreground">
                        {noteText || "(empty note)"}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="flex flex-col gap-4 lg:col-span-5 xl:col-span-5">
                  <PredictionPanelCard
                    kind="tp"
                    title="True Predictions"
                    count={tpCodes.length}
                    codes={tpCodes}
                    maps={m}
                    empty="No true positives at this iteration."
                    scrollMinH={Math.max(118, Math.floor(heights.list * 0.36))}
                  />
                  <PredictionPanelCard
                    kind="fp"
                    title="False Predictions"
                    count={fpCodes.length}
                    codes={fpCodes}
                    maps={m}
                    empty="No false positives."
                    scrollMinH={Math.max(118, Math.floor(heights.list * 0.36))}
                  />
                  <PredictionPanelCard
                    kind="fn"
                    title="Missed Predictions"
                    count={fnCodes.length}
                    codes={fnCodes}
                    maps={m}
                    empty="No false negatives."
                    scrollMinH={Math.max(118, Math.floor(heights.list * 0.36))}
                  />
                </div>
              </div>

              <div className="mt-6 lg:mt-8">
                <PredictionPanelCard
                  kind="gt"
                  title="Ground truth"
                  count={gtCodesSorted.length}
                  codes={gtCodesSorted}
                  maps={m}
                  empty="No ground-truth codes parsed for this row."
                  scrollMinH={Math.max(160, heights.list)}
                />
              </div>

              <div
                className={cn(
                  "animate-in fade-in-0 slide-in-from-bottom-2 rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm duration-500",
                )}
              >
                <div className="mb-4">
                  <h3 className="text-lg font-semibold tracking-tight">
                    Agent instructions
                  </h3>
                  <p className="text-xs text-muted-foreground">
                    Verifier bullets parsed from the think-block for each
                    timestep.
                  </p>
                </div>
                <div className="flex rounded-2xl bg-slate-100/90 p-1 ring-1 ring-slate-200/60">
                  {tabBtn("new", "New this iteration", newInstructions.length)}
                  {tabBtn(
                    "accumulated",
                    "All accumulated",
                    cumulativeInstructions.length,
                  )}
                </div>
                <div
                  key={instrTab}
                  className="animate-in fade-in-0 zoom-in-95 mt-6 duration-200"
                >
                  {instrTab === "new" ? (
                    <ul className="space-y-2.5 rounded-xl border border-slate-100 bg-slate-50/40 p-4">
                      {newInstructions.length === 0 ? (
                        <li className="text-sm text-muted-foreground">
                          (none for this iteration)
                        </li>
                      ) : (
                        newInstructions.map((t, i) => (
                          <li key={`${i}-${t.slice(0, 28)}`} className="flex gap-2 text-sm">
                            <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-sky-400" />
                            {t}
                          </li>
                        ))
                      )}
                    </ul>
                  ) : (
                    <ul className="space-y-2.5 rounded-xl border border-slate-100 bg-slate-50/40 p-4">
                      {cumulativeInstructions.length === 0 ? (
                        <li className="text-sm text-muted-foreground">
                          (none yet through this timestep)
                        </li>
                      ) : (
                        cumulativeInstructions.map((t, i) => (
                          <li key={`${i}-${t.slice(0, 28)}`} className="flex gap-2 text-sm">
                            <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-amber-400/90" />
                            {t}
                          </li>
                        ))
                      )}
                    </ul>
                  )}
                </div>
              </div>

              <p className="text-center text-xs text-muted-foreground">
                Tip: use ← / → arrow keys to step through iterations.
              </p>
            </>
          )}
        </section>

        <Collapsible open={diagOpen} onOpenChange={setDiagOpen} className="mt-12">
          <Card className="border-slate-200/90 shadow-sm">
            <CardHeader className="flex-row items-center justify-between space-y-0 pb-3">
              <div>
                <CardTitle className="text-base">Run-level diagnostics</CardTitle>
                <CardDescription className="text-xs">
                  Metrics, summary JSON, and output log tail.
                </CardDescription>
              </div>
              <CollapsibleTrigger
                className={cn(
                  buttonVariants({ variant: "outline", size: "sm" }),
                  "gap-1 rounded-xl",
                )}
              >
                <span className="flex items-center gap-1">
                  {diagOpen ? "Hide" : "Show"}
                  <ChevronDown
                    className={cn(
                      "size-4 transition-transform duration-200",
                      diagOpen && "rotate-180",
                    )}
                  />
                </span>
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="space-y-4 border-t border-slate-100 pt-4">
                {diag ? (
                  <>
                    {diag.chart && diag.chart.cols.length > 0 ? (
                      <div className="h-[260px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={chartData}>
                            <XAxis dataKey="step" />
                            <YAxis />
                            <RechartsTooltip />
                            <Legend />
                            {diag.chart.cols.map((c, i) => (
                              <Line
                                key={c}
                                type="monotone"
                                dataKey={c}
                                dot={false}
                                stroke={
                                  [
                                    "#2563eb",
                                    "#16a34a",
                                    "#9333ea",
                                    "#ea580c",
                                    "#0891b2",
                                    "#c026d3",
                                  ][i % 6]
                                }
                              />
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    ) : null}
                    <div>
                      <p className="mb-2 text-xs font-semibold uppercase text-muted-foreground">
                        wandb-summary
                      </p>
                      <pre className="max-h-[280px] overflow-auto rounded-xl border bg-muted/40 p-3 font-mono text-[11px] leading-relaxed">
                        {JSON.stringify(diag.summary, null, 2)}
                      </pre>
                    </div>
                    <div>
                      <p className="mb-2 text-xs font-semibold uppercase text-muted-foreground">
                        Output log (tail)
                      </p>
                      <pre className="max-h-[320px] overflow-auto rounded-xl border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-100">
                        {diag.logTail || "(empty)"}
                      </pre>
                    </div>
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground">Loading…</p>
                )}
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      </div>
    </div>
  );
}
