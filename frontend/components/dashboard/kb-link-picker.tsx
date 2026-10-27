"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { BookOpenCheck, ExternalLink, Link2, Search } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, api } from "@/lib/api";
import type { KbSuggestion } from "@/lib/types";

function useDebounced<T>(value: T, ms = 300) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return debounced;
}

/** Published KB articles for the reply box: suggested for this ticket, or searched; one click inserts the link. */
export function KbLinkPicker({ ticketId, onInsert }: { ticketId: number; onInsert: (a: KbSuggestion) => void }) {
  const [q, setQ] = useState("");
  const query = useDebounced(q.trim());
  const searching = query.length >= 2;
  const { data, isFetching, error } = useQuery({
    queryKey: ["kb-search", searching ? query : `ticket-${ticketId}`],
    queryFn: () => api.kbSearch(searching ? { q: query } : { ticket_id: ticketId }),
  });

  return (
    <div className="space-y-2 rounded-lg border bg-muted/30 p-3">
      <div className="relative">
        <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search published KB articles…"
          aria-label="Search KB articles"
          className="h-8 pl-8 text-sm"
        />
      </div>
      <p className="text-[11px] font-medium text-muted-foreground">
        {searching ? `Results for "${query}"` : "Suggested for this ticket"}
      </p>
      {isFetching && !data ? (
        <Skeleton className="h-12 w-full" />
      ) : error ? (
        <p className="py-2 text-xs text-destructive">
          Couldn&apos;t search articles: {error instanceof ApiError && error.status === 404
            ? "the server doesn't have this feature yet (restart the backend)"
            : String(error)}
        </p>
      ) : !data?.length ? (
        <p className="py-2 text-xs text-muted-foreground">
          {searching
            ? "No published article matches. Try other words."
            : "No published article closely matches this ticket. Search above, or publish one from the Knowledge page."}
        </p>
      ) : (
        <ul className="space-y-1.5">
          {data.map((a) => (
            <li key={a.id} className="flex items-center gap-2 rounded-md bg-background p-2">
              <BookOpenCheck className="size-4 shrink-0 text-muted-foreground" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium" title={a.title}>{a.title}</p>
                <p className="text-[11px] text-muted-foreground">
                  {a.number}
                  {a.product_area && ` · ${a.product_area}`} · {Math.round(a.score * 100)}% match
                </p>
              </div>
              <a
                href={`/support/kb/${a.number}`}
                target="_blank"
                rel="noreferrer"
                aria-label={`Open ${a.number}`}
                className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              >
                <ExternalLink className="size-3.5" />
              </a>
              <Button size="xs" variant="outline" onClick={() => onInsert(a)}>
                <Link2 className="size-3" /> Insert link
              </Button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
