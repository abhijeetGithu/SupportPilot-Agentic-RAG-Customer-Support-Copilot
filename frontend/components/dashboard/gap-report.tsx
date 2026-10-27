"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { SearchX } from "lucide-react";

import { formatDate, NativeSelect } from "@/components/labels";
import { StatTile } from "@/components/dashboard/metrics-overview";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import type { GapTopic } from "@/lib/types";

const OUTCOME_LABEL = { answered: "answered", ticket: "needed a ticket", handoff: "handed to a person", error: "error" };

const pct = (n: number, d: number) => (d ? `${Math.round((100 * n) / d)}%` : "–");

/** Customer questions the docs don't cover well, grouped into topics: what to write next. */
export function GapReportView() {
  const [days, setDays] = useState(30);
  const { data, isLoading } = useQuery({ queryKey: ["gaps", days], queryFn: () => api.gaps(days) });

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 text-sm">
        <label htmlFor="gap-days" className="text-muted-foreground">Period</label>
        <NativeSelect id="gap-days" value={days} onChange={(e) => setDays(Number(e.target.value))} className="w-36">
          <option value={7}>Last 7 days</option>
          <option value={30}>Last 30 days</option>
          <option value={90}>Last 90 days</option>
        </NativeSelect>
      </div>

      {isLoading || !data ? (
        <Skeleton className="h-64 w-full" />
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-3">
            <StatTile label="Questions asked in chat" value={String(data.questions)} />
            <StatTile
              label="Not well covered by the docs"
              value={pct(data.gap_questions, data.questions)}
              note={`${data.gap_questions} questions: weak doc match or needed a person`}
            />
            <StatTile
              label="Ended in a ticket or handoff"
              value={pct(data.ended_in_ticket_or_handoff, data.questions)}
              note={`${data.ended_in_ticket_or_handoff} conversations`}
            />
          </div>

          {data.topics.length === 0 ? (
            <div className="rounded-xl border border-dashed p-10 text-center text-sm text-muted-foreground">
              <SearchX className="mx-auto mb-2 size-6" />
              No gaps in this period. Questions show up here when the best doc match is weak or the chat ended with a
              ticket or a person.
            </div>
          ) : (
            <div className="rounded-xl border">
              <div className="border-b px-4 py-3">
                <h2 className="text-sm font-semibold">Topics customers ask about that the docs don&apos;t answer</h2>
                <p className="text-xs text-muted-foreground">
                  Similar questions grouped together, most frequent first. Write a doc or KB article for the top ones.
                </p>
              </div>
              <ol className="divide-y">
                {data.topics.map((t) => (
                  <TopicRow key={t.label + t.last_seen} topic={t} max={data.topics[0].count} />
                ))}
              </ol>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function TopicRow({ topic, max }: { topic: GapTopic; max: number }) {
  const outcomes = Object.entries(topic.outcomes) as [keyof typeof OUTCOME_LABEL, number][];
  return (
    <li className="space-y-2 px-4 py-3">
      <div className="flex items-start gap-3">
        <p className="min-w-0 flex-1 text-sm font-medium">{topic.label}</p>
        <div className="flex w-40 shrink-0 items-center gap-2" title={`${topic.count} questions`}>
          <div className="h-2 flex-1 rounded-full bg-muted">
            <div
              className="h-2 rounded-full"
              style={{ width: `${(100 * topic.count) / max}%`, background: "var(--viz-accent)" }}
            />
          </div>
          <span className="w-16 text-right text-xs tabular-nums">
            {topic.count} {topic.count === 1 ? "question" : "questions"}
          </span>
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        {outcomes.map(([k, n]) => `${n} ${OUTCOME_LABEL[k] ?? k}`).join(" · ")}
        {topic.avg_coverage !== null && <> · best doc match {Math.round(topic.avg_coverage * 100)}%</>}
        {" · "}last asked {formatDate(topic.last_seen)}
        {topic.closest_doc && (
          <>
            {" · "}closest doc:{" "}
            <a href={topic.closest_doc.url} target="_blank" rel="noreferrer" className="underline underline-offset-2">
              {topic.closest_doc.title}
            </a>
          </>
        )}
      </p>
      {topic.examples.length > 1 && (
        <details className="text-xs">
          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
            How customers asked ({topic.examples.length})
          </summary>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {topic.examples.map((q) => (
              <li key={q}>{q}</li>
            ))}
          </ul>
        </details>
      )}
    </li>
  );
}
