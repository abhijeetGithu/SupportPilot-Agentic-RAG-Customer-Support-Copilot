"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Headset, User } from "lucide-react";
import { toast } from "sonner";

import { SentFileChips } from "@/components/attachments";
import { Markdown } from "@/components/chat/chat-message";
import { StatusBadge, formatDate } from "@/components/labels";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, api } from "@/lib/api";
import { cn } from "@/lib/utils";

/** The URL (?number=&email=) is the source of truth, so links from the contact form open the ticket directly. */
export function TicketTracker() {
  const params = useSearchParams();
  const router = useRouter();
  const qc = useQueryClient();
  const qNumber = params.get("number") ?? "";
  const qEmail = params.get("email") ?? "";
  const [number, setNumber] = useState(qNumber);
  const [email, setEmail] = useState(qEmail);
  const [reply, setReply] = useState("");

  const key = ["customer-ticket", qNumber, qEmail];
  const { data: ticket, error, isFetching } = useQuery({
    queryKey: key,
    queryFn: () => api.lookupTicket(qNumber, qEmail),
    enabled: !!qNumber && !!qEmail,
    retry: false,
  });
  const send = useMutation({
    mutationFn: () => api.customerReply(qNumber, qEmail, reply),
    onSuccess: (view) => {
      qc.setQueryData(key, view);
      setReply("");
    },
    onError: (e) => toast.error("Couldn’t send your reply", { description: String(e) }),
  });

  return (
    <div className="space-y-6">
      <form
        className="grid gap-3 rounded-xl border p-5 sm:grid-cols-[1fr_1.5fr_auto] sm:items-end"
        onSubmit={(e) => {
          e.preventDefault();
          router.replace(`/support/tickets?number=${encodeURIComponent(number.trim())}&email=${encodeURIComponent(email.trim())}`);
        }}
      >
        <div className="space-y-1.5">
          <Label htmlFor="number">Ticket number</Label>
          <Input id="number" required placeholder="SP-00042" value={number} onChange={(e) => setNumber(e.target.value)} />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="email">Email used for the ticket</Label>
          <Input id="email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </div>
        <Button type="submit" disabled={isFetching}>
          {isFetching ? "Looking up…" : "Find ticket"}
        </Button>
      </form>

      {error && (
        <p className="text-sm text-destructive">
          {error instanceof ApiError && error.status === 404 ? "No ticket with that number for this email." : String(error)}
        </p>
      )}

      {ticket && (
        <div className="rounded-xl border">
          <div className="flex flex-wrap items-center gap-3 border-b p-5">
            <div className="min-w-0 flex-1">
              <p className="font-mono text-xs text-muted-foreground">{ticket.number}</p>
              <h2 className="truncate text-lg font-semibold">{ticket.subject}</h2>
              <p className="text-xs text-muted-foreground">
                Opened {formatDate(ticket.created_at)} · updated {formatDate(ticket.updated_at)}
              </p>
            </div>
            <StatusBadge status={ticket.status} />
          </div>

          {ticket.attachments.length > 0 && (
            <div className="border-b px-5 py-3">
              <p className="mb-1.5 text-xs text-muted-foreground">Files you attached (visible to the support team)</p>
              <SentFileChips files={ticket.attachments} />
            </div>
          )}

          <ol className="space-y-4 p-5">
            {ticket.messages.length === 0 && <li className="text-sm text-muted-foreground">No messages yet.</li>}
            {ticket.messages.map((m) => {
              const staff = m.author_type === "staff";
              return (
                <li key={m.id} className="flex gap-3">
                  <div
                    className={cn(
                      "flex size-8 shrink-0 items-center justify-center rounded-full",
                      staff ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300" : "bg-muted",
                    )}
                  >
                    {staff ? <Headset className="size-4" /> : <User className="size-4" />}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-xs text-muted-foreground">
                      <span className="font-medium text-foreground">{staff ? "Support team" : "You"}</span> ·{" "}
                      {formatDate(m.created_at)}
                    </p>
                    {staff ? (
                      <div className="mt-1"><Markdown text={m.body} /></div>
                    ) : (
                      <p className="mt-1 whitespace-pre-wrap text-sm">{m.body}</p>
                    )}
                  </div>
                </li>
              );
            })}
          </ol>

          {ticket.status !== "closed" && (
            <form
              className="space-y-2 border-t p-5"
              onSubmit={(e) => {
                e.preventDefault();
                send.mutate();
              }}
            >
              <Label htmlFor="reply">Add a reply</Label>
              <Textarea id="reply" rows={3} required value={reply} onChange={(e) => setReply(e.target.value)} />
              <div className="flex justify-end">
                <Button type="submit" disabled={send.isPending || !reply.trim()}>
                  Send reply
                </Button>
              </div>
            </form>
          )}
        </div>
      )}
    </div>
  );
}
