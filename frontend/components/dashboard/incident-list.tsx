"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { Activity, CheckCircle2 } from "lucide-react";

import { formatDate } from "@/components/labels";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import type { Incident } from "@/lib/types";
import { cn } from "@/lib/utils";

export function IncidentStatusBadge({ status }: { status: Incident["status"] }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium",
        status === "active"
          ? "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300"
          : "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
      )}
    >
      {status === "active" ? <Activity className="size-3" /> : <CheckCircle2 className="size-3" />}
      {status === "active" ? "Active" : "Resolved"}
    </span>
  );
}

/** Detected incidents, active first. */
export function IncidentList() {
  const { data, isLoading } = useQuery({ queryKey: ["incidents"], queryFn: () => api.incidents(), refetchInterval: 15000 });
  if (isLoading) return <Skeleton className="h-40 w-full rounded-xl" />;
  if (!data?.length)
    return (
      <div className="rounded-xl border border-dashed p-10 text-center text-sm text-muted-foreground">
        <Activity className="mx-auto mb-2 size-6" />
        No incidents. When several customers report the same problem within two hours, it shows up here.
      </div>
    );
  return (
    <ul className="space-y-3">
      {data.map((i) => (
        <li key={i.id}>
          <Link
            href={`/dashboard/incidents/${i.id}`}
            className={cn(
              "block rounded-xl border p-4 transition-colors hover:bg-muted/40",
              i.status === "active" && "border-red-200 dark:border-red-900",
            )}
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-xs text-muted-foreground">{i.number}</span>
              <IncidentStatusBadge status={i.status} />
              {i.status === "active" && !i.public_note && (
                <span className="text-xs text-amber-700 dark:text-amber-400">No customer notice yet</span>
              )}
            </div>
            <p className="mt-1 font-medium">{i.title}</p>
            {i.summary && <p className="mt-0.5 line-clamp-2 text-sm text-muted-foreground">{i.summary}</p>}
            <p className="mt-2 text-xs text-muted-foreground">
              {i.customers} customers · {i.open_tickets} open of {i.tickets} tickets · first report{" "}
              {formatDate(i.first_seen)} · latest {formatDate(i.last_seen)}
              {i.resolved_at && <> · resolved {formatDate(i.resolved_at)} by {i.resolved_by}</>}
            </p>
          </Link>
        </li>
      ))}
    </ul>
  );
}
