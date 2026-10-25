"use client";

import Link from "next/link";
import { useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { CircleAlert, Search } from "lucide-react";

import { CATEGORIES, NativeSelect, PRIORITIES, PriorityBadge, STATUSES, StatusBadge, formatDate } from "@/components/labels";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api } from "@/lib/api";

const PAGE_SIZE = 20;

export function TicketInbox() {
  const [filters, setFilters] = useState({ status: "", priority: "", category: "", needs_review: "", q: "" });
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);

  const { data, isPending, isFetching, error } = useQuery({
    queryKey: ["tickets", filters, page],
    queryFn: () =>
      api.tickets({ ...filters, needs_review: filters.needs_review || undefined, page, page_size: PAGE_SIZE }),
    placeholderData: keepPreviousData,
    refetchInterval: 15_000,
  });

  const setFilter = (k: keyof typeof filters) => (e: React.ChangeEvent<HTMLSelectElement>) => {
    setFilters((f) => ({ ...f, [k]: e.target.value }));
    setPage(1);
  };
  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <form
          className="relative min-w-60 flex-1"
          onSubmit={(e) => {
            e.preventDefault();
            setFilters((f) => ({ ...f, q: search }));
            setPage(1);
          }}
        >
          <Search className="absolute top-2.5 left-2.5 size-4 text-muted-foreground" />
          <Input className="pl-8" placeholder="Search subject, description or email" value={search} onChange={(e) => setSearch(e.target.value)} />
        </form>
        <NativeSelect aria-label="Status" value={filters.status} onChange={setFilter("status")}>
          <option value="">All statuses</option>
          {STATUSES.map((s) => <option key={s} value={s}>{s.replace("_", " ")}</option>)}
        </NativeSelect>
        <NativeSelect aria-label="Priority" value={filters.priority} onChange={setFilter("priority")}>
          <option value="">All priorities</option>
          {PRIORITIES.map((p) => <option key={p} value={p}>{p}</option>)}
        </NativeSelect>
        <NativeSelect aria-label="Category" value={filters.category} onChange={setFilter("category")}>
          <option value="">All categories</option>
          {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
        </NativeSelect>
        <NativeSelect aria-label="Review" value={filters.needs_review} onChange={setFilter("needs_review")}>
          <option value="">Any review state</option>
          <option value="true">Needs review</option>
          <option value="false">Reviewed / confident</option>
        </NativeSelect>
      </div>

      {error && <p className="text-sm text-destructive">Couldn’t load tickets: {String(error)}</p>}

      <div className="rounded-xl border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-28">Ticket</TableHead>
              <TableHead>Subject</TableHead>
              <TableHead>Category</TableHead>
              <TableHead>Priority</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Source</TableHead>
              <TableHead className="text-right">Created</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isPending &&
              Array.from({ length: 5 }).map((_, i) => (
                <TableRow key={i}>
                  <TableCell colSpan={7}><Skeleton className="h-5 w-full" /></TableCell>
                </TableRow>
              ))}
            {data?.items.length === 0 && (
              <TableRow>
                <TableCell colSpan={7} className="py-10 text-center text-sm text-muted-foreground">
                  No tickets match these filters.
                </TableCell>
              </TableRow>
            )}
            {data?.items.map((t) => (
              <TableRow key={t.id} className="cursor-pointer">
                <TableCell className="font-mono text-xs">
                  <Link href={`/dashboard/tickets/${t.id}`} className="hover:underline">{t.number}</Link>
                </TableCell>
                <TableCell className="max-w-80">
                  <Link href={`/dashboard/tickets/${t.id}`} className="block truncate font-medium hover:underline">
                    {t.subject}
                  </Link>
                  <span className="block truncate text-xs text-muted-foreground">{t.customer_email}</span>
                </TableCell>
                <TableCell className="text-sm">
                  <span className="flex items-center gap-1.5">
                    {t.needs_review && (
                      <CircleAlert className="size-3.5 text-amber-600" aria-label="Needs review" />
                    )}
                    {t.category ?? "—"}
                  </span>
                  {t.confidence != null && (
                    <span className="text-xs text-muted-foreground">{Math.round(t.confidence * 100)}% confident</span>
                  )}
                </TableCell>
                <TableCell><PriorityBadge priority={t.priority} /></TableCell>
                <TableCell><StatusBadge status={t.status} /></TableCell>
                <TableCell className="text-xs capitalize text-muted-foreground">{t.source}</TableCell>
                <TableCell className="text-right text-xs text-muted-foreground">{formatDate(t.created_at)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>
          {data ? `${data.total} ticket${data.total === 1 ? "" : "s"}` : ""} {isFetching && !isPending && "· refreshing…"}
        </span>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>Previous</Button>
          <span>Page {page} of {pages}</span>
          <Button variant="outline" size="sm" disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>Next</Button>
        </div>
      </div>
    </div>
  );
}
