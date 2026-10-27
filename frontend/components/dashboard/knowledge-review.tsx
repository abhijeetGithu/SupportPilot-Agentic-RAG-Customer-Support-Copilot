"use client";

import Link from "next/link";
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BookOpenCheck, ExternalLink, Eye, Pencil, Sparkles, TriangleAlert } from "lucide-react";
import { toast } from "sonner";

import { Markdown } from "@/components/chat/chat-message";
import { formatDate } from "@/components/labels";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/api";
import type { KbArticle, KbStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

const TABS: { status: KbStatus; label: string; empty: string }[] = [
  {
    status: "draft",
    label: "To review",
    empty: "No drafts. When you resolve a ticket that has a support reply, the AI drafts an article here.",
  },
  { status: "published", label: "Published", empty: "Nothing published yet." },
  { status: "rejected", label: "Rejected", empty: "No rejected drafts." },
];

/** Markdown plus heading styles for article bodies (## Problem, ## Resolution...). */
export function ArticleBody({ text }: { text: string }) {
  return (
    <div className="[&_h2]:mt-4 [&_h2]:mb-1 [&_h2]:text-sm [&_h2]:font-semibold [&_h2:first-child]:mt-0">
      <Markdown text={text} />
    </div>
  );
}

export function KnowledgeReview() {
  const [tab, setTab] = useState<KbStatus>("draft");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const { data: articles, isLoading } = useQuery({ queryKey: ["kb", tab], queryFn: () => api.kbArticles({ status: tab }) });
  const selected = articles?.find((a) => a.id === selectedId) ?? articles?.[0];
  const empty = TABS.find((t) => t.status === tab)!.empty;

  return (
    <div className="space-y-4">
      <div className="inline-flex rounded-lg border bg-muted/40 p-0.5" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.status}
            role="tab"
            aria-selected={tab === t.status}
            onClick={() => {
              setTab(t.status);
              setSelectedId(null);
            }}
            className={cn(
              "rounded-md px-3 py-1.5 text-sm transition-colors",
              tab === t.status ? "bg-background font-medium shadow-sm" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {isLoading ? (
        <Skeleton className="h-64 w-full" />
      ) : !articles?.length ? (
        <div className="rounded-xl border border-dashed p-10 text-center text-sm text-muted-foreground">
          <BookOpenCheck className="mx-auto mb-2 size-6" />
          {empty}
        </div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
          <ul className="space-y-1.5">
            {articles.map((a) => (
              <li key={a.id}>
                <button
                  onClick={() => setSelectedId(a.id)}
                  className={cn(
                    "w-full rounded-lg border p-3 text-left transition-colors",
                    selected?.id === a.id ? "border-primary bg-primary/5" : "hover:bg-muted/50",
                  )}
                >
                  <p className="line-clamp-2 text-sm font-medium">{a.title}</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {a.number}
                    {a.source_ticket_number && ` · from ${a.source_ticket_number}`} · {formatDate(a.created_at)}
                  </p>
                  {a.status === "published" && (
                    <p className="mt-1 text-xs text-muted-foreground">
                      Linked in {a.linked_in_replies} {a.linked_in_replies === 1 ? "reply" : "replies"} by agents
                    </p>
                  )}
                  {!a.generalizable && (
                    <p className="mt-1 flex items-center gap-1 text-xs text-amber-700 dark:text-amber-400">
                      <TriangleAlert className="size-3" /> May be specific to one customer
                    </p>
                  )}
                </button>
              </li>
            ))}
          </ul>
          {selected && <ArticleEditor key={`${selected.id}-${selected.updated_at}`} article={selected} />}
        </div>
      )}
    </div>
  );
}

function ArticleEditor({ article }: { article: KbArticle }) {
  const qc = useQueryClient();
  const [title, setTitle] = useState(article.title);
  const [area, setArea] = useState(article.product_area);
  const [keywords, setKeywords] = useState(article.keywords.join(", "));
  const [body, setBody] = useState(article.body);
  const [preview, setPreview] = useState(true);
  const changes = {
    title,
    product_area: area,
    keywords: keywords.split(",").map((k) => k.trim()).filter(Boolean),
    body,
  };
  const dirty =
    title !== article.title || area !== article.product_area || body !== article.body ||
    changes.keywords.join(",") !== article.keywords.join(",");

  const refresh = () => qc.invalidateQueries({ queryKey: ["kb"] });
  const save = useMutation({
    mutationFn: () => api.updateKb(article.id, changes),
    onSuccess: () => {
      toast.success(article.status === "published" ? "Saved and re-indexed" : "Draft saved");
      refresh();
    },
    onError: (e) => toast.error("Couldn't save", { description: String(e) }),
  });
  const act = useMutation({
    mutationFn: async (action: "publish" | "unpublish" | "reject") => {
      if (dirty && action === "publish") await api.updateKb(article.id, changes);
      return api.kbAction(article.id, action);
    },
    onSuccess: (a, action) => {
      toast.success(
        action === "publish"
          ? `${a.number} published: the assistant can now answer with it`
          : action === "unpublish"
            ? `${a.number} moved back to drafts`
            : `${a.number} rejected`,
      );
      refresh();
    },
    onError: (e) => toast.error("Action failed", { description: String(e) }),
  });
  const busy = save.isPending || act.isPending;
  const cov = article.doc_coverage;

  return (
    <div className="space-y-4 rounded-xl border p-5">
      <div className="flex flex-wrap items-start gap-2">
        <div className="min-w-0 flex-1">
          <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Sparkles className="size-3.5" /> {article.number} · drafted by {article.model ?? "AI"}
            {article.source_ticket_id && (
              <>
                {" "}from{" "}
                <Link href={`/dashboard/tickets/${article.source_ticket_id}`} className="underline underline-offset-2">
                  {article.source_ticket_number}
                </Link>
              </>
            )}
            {article.reviewed_by && <> · reviewed by {article.reviewed_by}</>}
          </p>
        </div>
        {article.status === "published" && (
          <a href={`/support/kb/${article.number}`} target="_blank" rel="noreferrer"
             className={buttonVariants({ variant: "outline", size: "sm" })}>
            <ExternalLink className="size-3.5" /> Public page
          </a>
        )}
      </div>

      <div className="grid gap-2 rounded-lg bg-muted/50 p-3 text-xs sm:grid-cols-2">
        <p>
          <span className="font-medium">Reusable?</span>{" "}
          {article.generalizable ? "Yes" : "Probably not"}
          {article.draft_note && <span className="text-muted-foreground">: {article.draft_note}</span>}
        </p>
        <p>
          <span className="font-medium">Closest existing doc:</span>{" "}
          {cov ? (
            <>
              <a href={cov.url} target="_blank" rel="noreferrer" className="underline underline-offset-2">{cov.title}</a>
              <span className="text-muted-foreground"> ({Math.round(cov.score * 100)}% similar). Check it doesn&apos;t already explain this fix.</span>
            </>
          ) : (
            <span className="text-muted-foreground">unknown</span>
          )}
        </p>
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="kb-title">Title</Label>
        <Input id="kb-title" value={title} onChange={(e) => setTitle(e.target.value)} />
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor="kb-area">Product area</Label>
          <Input id="kb-area" value={area} onChange={(e) => setArea(e.target.value)} />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="kb-keywords">Search keywords (comma-separated)</Label>
          <Input id="kb-keywords" value={keywords} onChange={(e) => setKeywords(e.target.value)} />
        </div>
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <Label htmlFor="kb-body">Article</Label>
          <Button variant="ghost" size="sm" onClick={() => setPreview((p) => !p)}>
            {preview ? <><Pencil className="size-3.5" /> Edit</> : <><Eye className="size-3.5" /> Preview</>}
          </Button>
        </div>
        {preview ? (
          <div className="rounded-lg border p-4"><ArticleBody text={body} /></div>
        ) : (
          <Textarea id="kb-body" rows={16} value={body} onChange={(e) => setBody(e.target.value)} className="font-mono text-xs" />
        )}
      </div>

      <div className="flex flex-wrap justify-end gap-2 border-t pt-4">
        {dirty && (
          <Button variant="outline" disabled={busy} onClick={() => save.mutate()}>
            Save changes
          </Button>
        )}
        {article.status !== "rejected" && (
          <Button variant="outline" disabled={busy} onClick={() => act.mutate("reject")}>
            Reject
          </Button>
        )}
        {article.status === "published" ? (
          <Button variant="outline" disabled={busy} onClick={() => act.mutate("unpublish")}>
            Unpublish
          </Button>
        ) : (
          <Button disabled={busy} onClick={() => act.mutate("publish")}>
            <BookOpenCheck className="size-4" /> {dirty ? "Save and publish" : "Publish"}
          </Button>
        )}
      </div>
    </div>
  );
}
