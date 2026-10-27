"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, ArrowLeft, BookOpenCheck, Bot, CircleAlert, Headset, Sparkles, User } from "lucide-react";
import { toast } from "sonner";

import { Markdown } from "@/components/chat/chat-message";
import { KbLinkPicker } from "@/components/dashboard/kb-link-picker";
import { TicketAttachments } from "@/components/dashboard/ticket-attachments";
import { CATEGORIES, NativeSelect, PRIORITIES, PriorityBadge, STATUSES, StatusBadge, formatDate } from "@/components/labels";
import { Button, buttonVariants } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/api";
import type { Citation, KbSuggestion, Ticket, TicketDetail as Detail } from "@/lib/types";
import { cn } from "@/lib/utils";

const AUTHOR = {
  customer: { label: "Customer", icon: User, style: "bg-muted" },
  ai: { label: "SupportPilot (AI)", icon: Bot, style: "bg-sky-100 text-sky-700 dark:bg-sky-950 dark:text-sky-300" },
  staff: { label: "Support team", icon: Headset, style: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300" },
} as const;

const EVENT_LABEL: Record<string, string> = {
  created: "Ticket created",
  classified: "Classified",
  reclassified: "Reclassified",
  status_changed: "Status changed",
  assigned: "Assigned",
  escalated: "Escalated to a human",
  reply: "Reply",
  duplicate_merged: "Duplicate report merged",
  kb_linked: "KB article linked",
  incident_opened: "Incident detected",
  incident_linked: "Linked to incident",
  incident_update: "Incident update sent",
  incident_unlinked: "Removed from incident",
  attachments_added: "Files attached",
};

function describeEvent(type: string, payload: Record<string, unknown> | null) {
  if (!payload) return "";
  if (type === "classified")
    return `${payload.category} · ${payload.priority} (${Math.round(Number(payload.confidence) * 100)}% / ${Math.round(Number(payload.priority_confidence) * 100)}%, ${payload.model})`;
  if ("field" in payload) return `${payload.field}: ${payload.old ?? "—"} → ${payload.new}`;
  if (type === "escalated") return String(payload.reason ?? "");
  if (type === "created") return `via ${payload.source}`;
  if (type === "kb_linked") return (payload.articles as string[]).join(", ");
  if (type.startsWith("incident_")) return String(payload.incident ?? "");
  if (type === "attachments_added") return (payload.files as string[]).join(", ");
  return "";
}

export function TicketDetail() {
  const { id } = useParams<{ id: string }>();
  const ticketId = Number(id);
  const qc = useQueryClient();
  const { data, error } = useQuery({ queryKey: ["ticket", ticketId], queryFn: () => api.ticket(ticketId), refetchInterval: 10_000 });

  const refresh = (d: Detail) => {
    qc.setQueryData(["ticket", ticketId], d);
    void qc.invalidateQueries({ queryKey: ["tickets"] });
    void qc.invalidateQueries({ queryKey: ["escalations"] });
  };
  const update = useMutation({
    mutationFn: (changes: Partial<Pick<Ticket, "status" | "priority" | "category">>) => api.updateTicket(ticketId, changes),
    onSuccess: (d) => {
      refresh(d);
      toast.success("Ticket updated");
    },
    onError: (e) => toast.error("Update failed", { description: String(e) }),
  });

  if (error) return <p className="text-sm text-destructive">Couldn’t load ticket: {String(error)}</p>;
  if (!data) return <Skeleton className="h-96 w-full rounded-xl" />;
  const { ticket: t, messages, events, attachments, incident } = data;

  return (
    <div className="space-y-6">
      <div>
        <Link href="/dashboard/tickets" className={cn(buttonVariants({ variant: "ghost", size: "sm" }), "-ml-2 gap-1")}>
          <ArrowLeft className="size-4" /> Tickets
        </Link>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <span className="font-mono text-sm text-muted-foreground">{t.number}</span>
          <StatusBadge status={t.status} />
          <PriorityBadge priority={t.priority} />
        </div>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">{t.subject}</h1>
        <p className="text-sm text-muted-foreground">
          {t.customer_name} · {t.customer_email} · opened {formatDate(t.created_at)} via {t.source}
        </p>
      </div>

      {incident && (
        <Link
          href={`/dashboard/incidents/${incident.id}`}
          className={cn(
            "flex items-center gap-2 rounded-lg border px-3 py-2 text-sm",
            incident.status === "active"
              ? "border-red-200 bg-red-50 text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200"
              : "bg-muted/50",
          )}
        >
          <Activity className="size-4 shrink-0" />
          <span className="min-w-0 flex-1">
            Part of {incident.status === "active" ? "active" : "resolved"} incident{" "}
            <span className="font-medium">{incident.number}</span>: {incident.title}
          </span>
          <span className="text-xs underline underline-offset-2">Open incident</span>
        </Link>
      )}

      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <div className="space-y-6">
          <section className="rounded-xl border">
            <div className="border-b p-4">
              <p className="mb-1 text-xs font-medium text-muted-foreground">Description</p>
              <p className="whitespace-pre-wrap text-sm">{t.description}</p>
            </div>
            <TicketAttachments attachments={attachments} />
            <ol className="space-y-4 p-4">
              {messages.length === 0 && <li className="text-sm text-muted-foreground">No messages yet.</li>}
              {messages.map((m) => {
                const a = AUTHOR[m.author_type];
                return (
                  <li key={m.id} className="flex gap-3">
                    <div className={cn("flex size-8 shrink-0 items-center justify-center rounded-full", a.style)}>
                      <a.icon className="size-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-xs text-muted-foreground">
                        <span className="font-medium text-foreground">{a.label}</span> · {formatDate(m.created_at)}
                      </p>
                      {m.author_type === "customer" ? (
                        <p className="mt-1 whitespace-pre-wrap text-sm">{m.body}</p>
                      ) : (
                        <div className="mt-1"><Markdown text={m.body} /></div>
                      )}
                    </div>
                  </li>
                );
              })}
            </ol>
          </section>
          <ReplyComposer ticketId={ticketId} disabled={t.status === "closed"} onSent={refresh} />
        </div>

        <aside className="space-y-4">
          <section className="space-y-3 rounded-xl border p-4">
            {t.needs_review && (
              <div className="flex gap-2 rounded-lg bg-amber-50 p-3 text-xs text-amber-900 dark:bg-amber-950 dark:text-amber-200">
                <CircleAlert className="size-4 shrink-0" />
                <div className="space-y-2">
                  <p>The classifier wasn’t confident. Check the category and priority.</p>
                  <Button size="xs" variant="outline" onClick={() => update.mutate({ category: t.category ?? undefined })}>
                    Confirm classification
                  </Button>
                </div>
              </div>
            )}
            <Field label="Status">
              <NativeSelect value={t.status} disabled={update.isPending} onChange={(e) => update.mutate({ status: e.target.value as Ticket["status"] })}>
                {STATUSES.map((s) => <option key={s} value={s}>{s.replace("_", " ")}</option>)}
              </NativeSelect>
            </Field>
            <Field label="Priority" hint={t.priority_confidence != null ? `${Math.round(t.priority_confidence * 100)}% confident` : undefined}>
              <NativeSelect value={t.priority} disabled={update.isPending} onChange={(e) => update.mutate({ priority: e.target.value as Ticket["priority"] })}>
                {PRIORITIES.map((p) => <option key={p} value={p}>{p}</option>)}
              </NativeSelect>
            </Field>
            <Field label="Category" hint={t.confidence != null ? `${Math.round(t.confidence * 100)}% confident · ${t.classifier_model}` : undefined}>
              <NativeSelect value={t.category ?? ""} disabled={update.isPending} onChange={(e) => update.mutate({ category: e.target.value })}>
                {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
              </NativeSelect>
            </Field>
            <Field label="Assignee">
              <p className="text-sm">{t.assignee ?? "Unassigned"}</p>
            </Field>
          </section>

          <KnowledgeCard
            key={String(t.status === "resolved")} // remount on resolve: restarts the "drafting" window
            ticketId={ticketId}
            resolved={t.status === "resolved" || t.status === "closed"}
            hasStaffReply={messages.some((m) => m.author_type === "staff")}
          />

          <section className="rounded-xl border p-4">
            <p className="mb-3 text-xs font-medium text-muted-foreground">Activity</p>
            <ol className="space-y-3 border-l pl-4">
              {events.map((e) => (
                <li key={e.id} className="relative text-xs">
                  <span className="absolute top-1 -left-[21px] size-2 rounded-full bg-muted-foreground/40" />
                  <p className="font-medium">{EVENT_LABEL[e.type] ?? e.type}</p>
                  <p className="text-muted-foreground">{describeEvent(e.type, e.payload)}</p>
                  <p className="text-muted-foreground">{e.actor} · {formatDate(e.created_at)}</p>
                </li>
              ))}
            </ol>
          </section>
        </aside>
      </div>
    </div>
  );
}

/** The KB article learned from this ticket, or a button to draft one from the staff reply. */
function KnowledgeCard({ ticketId, resolved, hasStaffReply }: { ticketId: number; resolved: boolean; hasStaffReply: boolean }) {
  const qc = useQueryClient();
  // resolving starts a background draft (~10 s): poll for up to 45 s until it appears
  const [waiting, setWaiting] = useState(resolved && hasStaffReply);
  useEffect(() => {
    const id = setTimeout(() => setWaiting(false), 45_000);
    return () => clearTimeout(id);
  }, []);
  const { data: articles } = useQuery({
    queryKey: ["kb", "ticket", ticketId],
    queryFn: () => api.kbArticles({ ticket_id: ticketId }),
    refetchInterval: (q) => (waiting && !q.state.data?.length ? 3000 : false),
  });
  const draft = useMutation({
    mutationFn: () => api.draftKbFromTicket(ticketId),
    onSuccess: (a) => {
      toast.success(`${a.number} drafted`, { description: "Review it on the Knowledge page." });
      qc.invalidateQueries({ queryKey: ["kb"] });
    },
    onError: (e) => toast.error("Couldn't draft an article", { description: String(e) }),
  });
  const article = articles?.find((a) => a.status !== "rejected") ?? articles?.[0];
  if (!hasStaffReply && !article) return null;
  return (
    <section className="space-y-2 rounded-xl border p-4 text-sm">
      <p className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        <BookOpenCheck className="size-3.5" /> Knowledge base
      </p>
      {article ? (
        <>
          <p className="font-medium leading-snug">{article.title}</p>
          <p className="text-xs text-muted-foreground">
            {article.number} · {article.status === "draft" ? "waiting for review" : article.status}
          </p>
          <Link href="/dashboard/knowledge" className={buttonVariants({ variant: "outline", size: "sm" })}>
            {article.status === "draft" ? "Review article" : "Open in Knowledge"}
          </Link>
        </>
      ) : (
        <>
          <p className="text-xs text-muted-foreground">
            {waiting
              ? "Drafting an article from this fix…"
              : "Turn this fix into an article so the assistant can answer it next time."}
          </p>
          <Button size="sm" variant="outline" disabled={draft.isPending} onClick={() => draft.mutate()}>
            <Sparkles className="size-3.5" /> {draft.isPending ? "Drafting…" : "Draft KB article"}
          </Button>
        </>
      )}
    </section>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <Label className="text-xs text-muted-foreground">{label}</Label>
      <div className="[&>select]:w-full">{children}</div>
      {hint && <p className="text-[11px] text-muted-foreground">{hint}</p>}
    </div>
  );
}

/** Human in the loop: the AI drafts, staff edit and send. Nothing is sent automatically. */
function ReplyComposer({ ticketId, disabled, onSent }: { ticketId: number; disabled: boolean; onSent: (d: Detail) => void }) {
  const [body, setBody] = useState("");
  const [sources, setSources] = useState<Citation[]>([]);
  const [linking, setLinking] = useState(false);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const insertLink = (a: KbSuggestion) => {
    const link = `[${a.title}](${a.url})`;
    if (!body.trim()) {
      setBody(`This article walks through the fix: ${link}`);
      return;
    }
    const el = textarea.current;
    const start = el?.selectionStart ?? body.length;
    const end = el?.selectionEnd ?? body.length;
    const before = body.slice(0, start);
    const spaced = before && !/\s$/.test(before) ? ` ${link}` : link;
    setBody(before + spaced + body.slice(end));
    toast.success(`Linked ${a.number}`);
  };
  const suggest = useMutation({
    mutationFn: () => api.suggestReply(ticketId),
    onSuccess: (r) => {
      setBody(r.body);
      setSources(r.citations);
    },
    onError: (e) => toast.error("Couldn't draft a reply", { description: String(e) }),
  });
  const send = useMutation({
    mutationFn: () => api.staffReply(ticketId, body),
    onSuccess: (d) => {
      onSent(d);
      setBody("");
      setSources([]);
      toast.success("Reply sent");
    },
    onError: (e) => toast.error("Couldn't send", { description: String(e) }),
  });

  return (
    <section className="space-y-2 rounded-xl border p-4">
      <div className="flex items-center justify-between">
        <Label htmlFor="reply">Reply to customer</Label>
        <div className="flex gap-1.5">
          <Button variant={linking ? "secondary" : "outline"} size="sm" className="gap-1.5" disabled={disabled}
                  aria-expanded={linking} onClick={() => setLinking((v) => !v)}>
            <BookOpenCheck className="size-3.5" /> Link KB article
          </Button>
          <Button variant="outline" size="sm" className="gap-1.5" disabled={disabled || suggest.isPending} onClick={() => suggest.mutate()}>
            <Sparkles className="size-3.5" /> {suggest.isPending ? "Drafting…" : "Suggest reply"}
          </Button>
        </div>
      </div>
      {linking && !disabled && <KbLinkPicker ticketId={ticketId} onInsert={insertLink} />}
      <Textarea id="reply" ref={textarea} rows={8} value={body} onChange={(e) => setBody(e.target.value)} disabled={disabled} placeholder="Write a reply, or let the assistant draft one from the docs." />
      {sources.length > 0 && (
        <div className="flex flex-wrap gap-1.5 text-[11px]">
          <span className="text-muted-foreground">Sources:</span>
          {sources.map((c) => (
            <a key={c.n} href={c.url} target="_blank" rel="noreferrer" className="rounded border px-1.5 py-0.5 hover:bg-muted">
              [{c.n}] {c.title}
            </a>
          ))}
        </div>
      )}
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">Check AI drafts before sending; you are the one replying.</p>
        <Button disabled={disabled || !body.trim() || send.isPending} onClick={() => send.mutate()}>
          {send.isPending ? "Sending…" : "Send reply"}
        </Button>
      </div>
    </section>
  );
}
