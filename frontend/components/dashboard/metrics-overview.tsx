"use client";

import { useQuery } from "@tanstack/react-query";
import { Bar, BarChart, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { PRIORITIES, STATUSES } from "@/components/labels";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

const fmt = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
const pct = (x: number) => `${Math.round(x * 100)}%`;

type Row = { label: string; value: number };

export function MetricsOverview() {
  const { data, isPending, isFetching, error } = useQuery({
    queryKey: ["metrics"],
    queryFn: api.metrics,
    refetchInterval: 15_000,
  });

  if (error) return <p className="text-sm text-destructive">Couldn’t load metrics: {String(error)}</p>;
  if (isPending || !data) return <Skeleton className="h-96 w-full rounded-xl" />;

  const total = Object.values(data.tickets_by_status).reduce((a, b) => a + b, 0);
  const active = ["open", "in_progress", "escalated"].reduce((a, s) => a + (data.tickets_by_status[s] ?? 0), 0);

  const byCategory: Row[] = Object.entries(data.tickets_by_category)
    .map(([label, value]) => ({ label, value }))
    .sort((a, b) => b.value - a.value);
  // ordinal scales keep their natural order, not sorted by size
  const byPriority: Row[] = PRIORITIES.map((p) => ({ label: p, value: data.tickets_by_priority[p] ?? 0 }));
  const byStatus: Row[] = STATUSES.map((s) => ({ label: s.replace("_", " "), value: data.tickets_by_status[s] ?? 0 }));

  return (
    // refetch keeps the frame: dim while refreshing, no skeleton flash
    <div className={cn("space-y-6 transition-opacity", isFetching && "opacity-70")}>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <StatTile label="Tickets" value={fmt.format(total)} />
        <StatTile label="Active" value={fmt.format(active)} note="open, in progress or escalated" />
        <StatTile label="Needs review" value={fmt.format(data.needs_review)} note="low classifier confidence" />
        <StatTile label="AI resolution rate" value={pct(data.ai_resolution_rate)} note="chats closed without a ticket" />
        <StatTile label="Chat sessions" value={fmt.format(data.chat_sessions)} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartCard title="Tickets by category" subtitle="Classifier-assigned team" rows={byCategory} total={total} className="lg:row-span-2" />
        <ChartCard title="Tickets by priority" subtitle="Urgent is set by safety rules" rows={byPriority} total={total} />
        <ChartCard title="Tickets by status" rows={byStatus} total={total} />
      </div>
    </div>
  );
}

export function StatTile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded-xl border p-4">
      <p className="text-sm text-muted-foreground">{label}</p>
      {/* proportional figures for standalone numbers */}
      <p className="mt-1 text-3xl font-semibold tracking-tight">{value}</p>
      {note && <p className="mt-1 text-xs text-muted-foreground">{note}</p>}
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  rows,
  total,
  className,
}: {
  title: string;
  subtitle?: string;
  rows: Row[];
  total: number;
  className?: string;
}) {
  const shown = rows.filter((r) => r.value > 0);
  const longest = Math.max(...rows.map((r) => r.label.length), 4);
  const BAR = 18; // thin bars (<= 24px), the rest of each 32px band is air
  const height = rows.length * 32 + 8;

  return (
    <section className={cn("rounded-xl border p-4", className)}>
      <h2 className="text-sm font-medium">{title}</h2>
      {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}

      {shown.length < 2 ? (
        // a one-bar chart says less than a sentence
        <p className="py-8 text-sm text-muted-foreground">
          {shown.length === 0
            ? "No tickets yet."
            : total === 1
              ? `The only ticket so far: ${shown[0].label}.`
              : `All ${total} tickets: ${shown[0].label}.`}
        </p>
      ) : (
        <div className="mt-3" style={{ height }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 40, bottom: 0, left: 0 }} barCategoryGap={0}>
              {/* every value is labelled at the bar tip, so the value axis and grid are omitted */}
              <XAxis type="number" hide domain={[0, "dataMax"]} />
              <YAxis
                type="category"
                dataKey="label"
                width={Math.min(220, longest * 7 + 8)}
                tickLine={false}
                axisLine={{ stroke: "var(--border)" }}
                tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                tickFormatter={(v: string) => v.charAt(0).toUpperCase() + v.slice(1)}
              />
              <Tooltip
                cursor={{ fill: "var(--muted)", opacity: 0.6 }}
                isAnimationActive={false}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const r = payload[0].payload as Row;
                  return (
                    <div className="rounded-md border bg-popover px-2.5 py-1.5 text-xs shadow-md">
                      {/* value leads, label follows */}
                      <p className="text-sm font-semibold text-popover-foreground">
                        {r.value} <span className="font-normal text-muted-foreground">· {total ? pct(r.value / total) : "0%"} of tickets</span>
                      </p>
                      <p className="text-muted-foreground capitalize">{r.label}</p>
                    </div>
                  );
                }}
              />
              <Bar
                dataKey="value"
                fill="var(--viz-accent)"
                barSize={BAR}
                radius={[0, 4, 4, 0]} // rounded data-end, square at the baseline
                activeBar={{ fillOpacity: 0.8 }}
                isAnimationActive={false}
              >
                <LabelList dataKey="value" position="right" offset={6} style={{ fill: "var(--foreground)", fontSize: 12 }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <details className="mt-2 text-xs">
        <summary className="cursor-pointer text-muted-foreground hover:text-foreground">View as table</summary>
        <table className="mt-2 w-full">
          <thead>
            <tr className="text-left text-muted-foreground">
              <th className="py-1 font-normal">Group</th>
              <th className="py-1 text-right font-normal">Tickets</th>
              <th className="py-1 text-right font-normal">Share</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {rows.map((r) => (
              <tr key={r.label} className="border-t">
                <td className="py-1 capitalize">{r.label}</td>
                <td className="py-1 text-right">{r.value}</td>
                <td className="py-1 text-right">{total ? pct(r.value / total) : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </section>
  );
}
