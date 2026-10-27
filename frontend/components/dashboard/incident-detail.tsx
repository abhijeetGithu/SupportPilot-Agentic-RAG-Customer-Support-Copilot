"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Megaphone, Sparkles, X } from "lucide-react";
import { toast } from "sonner";

import { IncidentStatusBadge } from "@/components/dashboard/incident-list";
import { StatusBadge, formatDate } from "@/components/labels";
import { Button, buttonVariants } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/api";
import type { Incident, IncidentDetail as Detail } from "@/lib/types";
import { cn } from "@/lib/utils";

function chance(p?: number) {
  if (p === undefined) return "unknown";
  return p < 0.0001 ? "under 0.01%" : `${(p * 100).toFixed(2)}%`;
}

/** Why the detector opened this incident, in plain words. */
function DetectionFacts({ incident }: { incident: Incident }) {
  const d = incident.detection;
  return (
    <p className="text-xs text-muted-foreground">
      Detected when {d.customers ?? "several"} different customers reported similar problems within{" "}
      {d.window_minutes ?? 120} minutes. Usually there are about {d.expected_per_window ?? 0} such reports in that
      time ({d.baseline_similar ?? 0} similar tickets in the previous week), so the chance of this by coincidence is{" "}
      {chance(d.p_value)}.
    </p>
  );
}

export function IncidentDetailView() {
  const { id } = useParams<{ id: string }>();
  const incidentId = Number(id);
  const qc = useQueryClient();
  const { data, error } = useQuery({ queryKey: ["incident", incidentId], queryFn: () => api.incident(incidentId) });
  const set = (d: Detail) => {
    qc.setQueryData(["incident", incidentId], d);
    qc.invalidateQueries({ queryKey: ["incidents"] });
  };
  const summarize = useMutation({
    mutationFn: () => api.summarizeIncident(incidentId),
    onSuccess: set,
    onError: (e) => toast.error("Couldn't summarize", { description: String(e) }),
  });
  // write an AI title/summary once, the first time staff open a new incident
  const asked = useRef(false);
  useEffect(() => {
    if (data && !data.incident.summary && data.incident.status === "active" && !asked.current) {
      asked.current = true;
      summarize.mutate();
    }
  }, [data, summarize]);

  if (error) return <p className="text-sm text-destructive">Couldn&apos;t load incident: {String(error)}</p>;
  if (!data) return <Skeleton className="h-96 w-full rounded-xl" />;
  const { incident: i, tickets } = data;
  const active = i.status === "active";
  const start = Date.parse(i.first_seen);

  return (
    <div className="space-y-6">
      <div>
        <Link href="/dashboard/incidents" className={cn(buttonVariants({ variant: "ghost", size: "sm" }), "-ml-2 gap-1")}>
          <ArrowLeft className="size-4" /> Incidents
        </Link>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <span className="font-mono text-sm text-muted-foreground">{i.number}</span>
          <IncidentStatusBadge status={i.status} />
        </div>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">{i.title}</h1>
        <div className="mt-1 flex flex-wrap items-start gap-2">
          <p className="min-w-0 flex-1 text-sm text-muted-foreground">
            {summarize.isPending ? "Summarizing the reports…" : i.summary || "No summary yet."}
          </p>
          <Button variant="ghost" size="sm" disabled={summarize.isPending} onClick={() => summarize.mutate()}>
            <Sparkles className="size-3.5" /> {i.summary ? "Re-summarize" : "Summarize"}
          </Button>
        </div>
        <p className="mt-1 text-[11px] text-muted-foreground">AI title and summary are for staff only; customers never see them.</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
        <section className="rounded-xl border">
          <div className="border-b p-4">
            <p className="text-sm font-medium">
              {i.customers} customers · {i.tickets} tickets · {i.open_tickets} open
            </p>
            <DetectionFacts incident={i} />
          </div>
          <ol className="divide-y">
            {tickets.map((t) => (
              <li key={t.id} className="flex items-start gap-3 p-4">
                <span className="w-16 shrink-0 pt-0.5 text-xs text-muted-foreground tabular-nums">
                  +{Math.max(0, Math.round((Date.parse(t.created_at) - start) / 60000))} min
                </span>
                <div className="min-w-0 flex-1">
                  <Link href={`/dashboard/tickets/${t.id}`} className="text-sm font-medium hover:underline">
                    {t.subject}
                  </Link>
                  <p className="text-xs text-muted-foreground">
                    {t.number} · {t.customer_email} · {formatDate(t.created_at)}
                  </p>
                </div>
                <StatusBadge status={t.status} />
                {active && <RemoveTicket incidentId={incidentId} ticketId={t.id} onDone={set} />}
              </li>
            ))}
          </ol>
        </section>

        <aside className="space-y-4">
          {active ? (
            <>
              <PublicNote incident={i} onSaved={set} />
              <SendUpdate incident={i} onSent={() => qc.invalidateQueries({ queryKey: ["incident", incidentId] })} />
              <Resolve incident={i} onDone={set} />
            </>
          ) : (
            <section className="rounded-xl border p-4 text-sm">
              Resolved {i.resolved_at && formatDate(i.resolved_at)} by {i.resolved_by}.
            </section>
          )}
        </aside>
      </div>
    </div>
  );
}

function PublicNote({ incident, onSaved }: { incident: Incident; onSaved: (d: Detail) => void }) {
  const [note, setNote] = useState(incident.public_note);
  const save = useMutation({
    mutationFn: () => api.updateIncident(incident.id, { public_note: note }),
    onSuccess: (d) => {
      onSaved(d);
      toast.success("Customer notice saved", { description: "The assistant and new tickets will use it." });
    },
    onError: (e) => toast.error("Couldn't save", { description: String(e) }),
  });
  return (
    <section className="space-y-2 rounded-xl border p-4">
      <Label htmlFor="note">Customer notice</Label>
      <p className="text-xs text-muted-foreground">
        The assistant tells customers who describe this problem, and new matching tickets get it as an automatic reply.
      </p>
      <Textarea id="note" rows={4} value={note} onChange={(e) => setNote(e.target.value)}
                placeholder="e.g. We're aware that Salesforce crawls are failing and are deploying a fix. Next update by 3 PM." />
      <div className="flex justify-end">
        <Button size="sm" disabled={save.isPending || note === incident.public_note} onClick={() => save.mutate()}>
          Save notice
        </Button>
      </div>
    </section>
  );
}

function SendUpdate({ incident, onSent }: { incident: Incident; onSent: () => void }) {
  const [body, setBody] = useState("");
  const send = useMutation({
    mutationFn: () => api.incidentUpdate(incident.id, body),
    onSuccess: (r) => {
      toast.success(`Update sent to ${r.sent_to} customer${r.sent_to === 1 ? "" : "s"}`);
      setBody("");
      onSent();
    },
    onError: (e) => toast.error("Couldn't send", { description: String(e) }),
  });
  return (
    <section className="space-y-2 rounded-xl border p-4">
      <Label htmlFor="update">Update everyone</Label>
      <Textarea id="update" rows={4} value={body} onChange={(e) => setBody(e.target.value)}
                placeholder="Posted as a reply on every open ticket in this incident." />
      <div className="flex justify-end">
        <Button size="sm" disabled={send.isPending || body.trim().length < 5 || incident.open_tickets === 0}
                onClick={() => send.mutate()}>
          <Megaphone className="size-3.5" /> Send to {incident.open_tickets} customer{incident.open_tickets === 1 ? "" : "s"}
        </Button>
      </div>
    </section>
  );
}

function Resolve({ incident, onDone }: { incident: Incident; onDone: (d: Detail) => void }) {
  const [message, setMessage] = useState("");
  const [closeTickets, setCloseTickets] = useState(true);
  const resolve = useMutation({
    mutationFn: () => api.resolveIncident(incident.id, message.trim() || null, closeTickets),
    onSuccess: (d) => {
      onDone(d);
      toast.success(`${incident.number} resolved`);
    },
    onError: (e) => toast.error("Couldn't resolve", { description: String(e) }),
  });
  return (
    <section className="space-y-2 rounded-xl border p-4">
      <Label htmlFor="resolve">Resolve incident</Label>
      <Textarea id="resolve" rows={3} value={message} onChange={(e) => setMessage(e.target.value)}
                placeholder="Optional final message to every open ticket, e.g. Fixed: crawls are running again." />
      <label className="flex items-center gap-2 text-xs">
        <input type="checkbox" checked={closeTickets} onChange={(e) => setCloseTickets(e.target.checked)} />
        Also resolve the {incident.open_tickets} open ticket{incident.open_tickets === 1 ? "" : "s"}
      </label>
      <div className="flex justify-end">
        <Button size="sm" variant="outline" disabled={resolve.isPending} onClick={() => resolve.mutate()}>
          Resolve
        </Button>
      </div>
    </section>
  );
}

function RemoveTicket({ incidentId, ticketId, onDone }: { incidentId: number; ticketId: number; onDone: (d: Detail) => void }) {
  const remove = useMutation({
    mutationFn: () => api.removeIncidentTicket(incidentId, ticketId),
    onSuccess: (d) => {
      onDone(d);
      toast.success("Ticket removed from the incident");
    },
    onError: (e) => toast.error("Couldn't remove", { description: String(e) }),
  });
  return (
    <Button variant="ghost" size="icon-sm" aria-label="Remove from incident" title="Not part of this incident"
            disabled={remove.isPending} onClick={() => remove.mutate()}>
      <X className="size-3.5" />
    </Button>
  );
}
