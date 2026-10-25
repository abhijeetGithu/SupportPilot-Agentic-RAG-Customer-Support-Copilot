"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Headset, MessageCircle, Paperclip, RotateCcw, SendHorizontal, Sparkles, X } from "lucide-react";
import { toast } from "sonner";

import { AttachMenu, PendingFileChips, useUploads } from "@/components/attachments";
import { ChatMessage, type UiMessage } from "@/components/chat/chat-message";
import { TicketDraftCard } from "@/components/chat/ticket-draft-card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, api } from "@/lib/api";
import type { ChatEvent, SessionMode, TicketDraft } from "@/lib/types";
import { cn } from "@/lib/utils";

const STORAGE_KEY = "supportpilot.chat";
const POLL_MS = 4000;
export const OPEN_CHAT_EVENT = "supportpilot:open-chat";

type Stored = { sessionId: string; name: string };
type Item = { key: string; msg: UiMessage };

const TOOL_STATUS: Record<string, (q?: string) => string> = {
  search_docs: (q) => (q ? `Searching the docs for "${q}"…` : "Searching the docs…"),
  create_ticket: () => "Preparing a ticket draft…",
  get_ticket_status: () => "Checking your ticket…",
  escalate_to_human: () => "Connecting you to the support team…",
};

let keySeq = 0;
const nextKey = () => `m${++keySeq}`;

/** Open the widget from anywhere, optionally sending a first question. */
export function openChat(question?: string) {
  window.dispatchEvent(new CustomEvent(OPEN_CHAT_EVENT, { detail: { question } }));
}

export function ChatWidget() {
  const [open, setOpen] = useState(false);
  // Safe to read localStorage on first render: everything that depends on it is inside the panel,
  // which only renders after the user opens it, so server and client HTML match.
  const [stored, setStored] = useState<Stored | null>(() => {
    if (typeof window === "undefined") return null;
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Stored) : null;
  });
  const [items, setItems] = useState<Item[]>([]);
  const [draft, setDraft] = useState<TicketDraft | null>(null);
  const [mode, setMode] = useState<SessionMode>("ai");
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState("");
  const [dragging, setDragging] = useState(false);
  const uploads = useUploads(stored?.sessionId);
  const canAttach = !!stored && !busy && !draft;
  const lastStaffId = useRef(0);
  const queued = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const push = useCallback((msg: UiMessage) => {
    const key = nextKey();
    setItems((xs) => [...xs, { key, msg }]);
    return key;
  }, []);

  const patchAi = useCallback((key: string, fn: (m: Extract<UiMessage, { kind: "ai" }>) => Partial<UiMessage>) => {
    setItems((xs) => xs.map((it) => (it.key === key && it.msg.kind === "ai" ? { key, msg: { ...it.msg, ...fn(it.msg) } as UiMessage } : it)));
  }, []);

  // ---------- restore the conversation ----------
  useEffect(() => {
    if (!stored) return;
    api
      .history(stored.sessionId)
      .then((h) => {
        setMode(h.mode);
        setDraft(h.pending_draft);
        const restored: Item[] = h.messages.map((m) => ({
          key: nextKey(),
          msg:
            m.role === "customer"
              ? { kind: "customer", text: m.content, attachments: m.attachments }
              : m.role === "staff"
                ? { kind: "staff", text: m.content, id: m.id ?? 0 }
                : { kind: "ai", text: m.content, citations: [] },
        }));
        lastStaffId.current = Math.max(0, ...h.messages.map((m) => (m.role === "staff" ? (m.id ?? 0) : 0)));
        setItems(restored);
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 404) resetSession(); // session no longer exists
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stored?.sessionId]);

  // ---------- open from other components ----------
  useEffect(() => {
    const handler = (e: Event) => {
      setOpen(true);
      const q = (e as CustomEvent<{ question?: string }>).detail?.question;
      if (q) queued.current = q;
    };
    window.addEventListener(OPEN_CHAT_EVENT, handler);
    return () => window.removeEventListener(OPEN_CHAT_EVENT, handler);
  }, []);

  // ---------- auto-scroll ----------
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [items, draft]);

  // ---------- human handoff: poll for staff replies ----------
  useEffect(() => {
    if (!stored || mode !== "human") return;
    const id = setInterval(async () => {
      try {
        const res = await api.staffMessages(stored.sessionId, lastStaffId.current);
        for (const m of res.messages) {
          lastStaffId.current = Math.max(lastStaffId.current, m.id ?? 0);
          push({ kind: "staff", text: m.content, id: m.id ?? 0 });
        }
        if (res.mode === "ai") {
          setMode("ai");
          push({ kind: "notice", tone: "success", text: `Ticket ${res.ticket_number} was resolved. The assistant is back.` });
        }
      } catch {
        /* transient network errors: try again on the next tick */
      }
    }, POLL_MS);
    return () => clearInterval(id);
  }, [stored, mode, push]);

  // ---------- streaming ----------
  const consume = useCallback(
    async (events: AsyncGenerator<ChatEvent>) => {
      const aiKey = push({ kind: "ai", text: "", citations: [], streaming: true });
      setBusy(true);
      try {
        for await (const e of events) {
          switch (e.type) {
            case "token":
              patchAi(aiKey, (m) => ({ text: m.text + e.text, status: undefined }));
              break;
            case "tool_start":
              patchAi(aiKey, () => ({ status: (TOOL_STATUS[e.tool] ?? (() => "Working…"))(e.query) }));
              break;
            case "tool_end":
              patchAi(aiKey, () => ({ status: undefined }));
              break;
            case "citations":
              patchAi(aiKey, () => ({ citations: e.citations }));
              break;
            case "ticket_draft":
              setDraft(e.draft);
              break;
            case "ticket_created":
              push({
                kind: "notice",
                tone: "success",
                text: e.duplicate
                  ? `Added to your open ticket ${e.number}`
                  : `Ticket ${e.number} created · ${e.category} · ${e.priority} priority`,
              });
              break;
            case "handoff":
              setMode("human");
              push({ kind: "notice", tone: "info", text: `Ticket ${e.number}: a support team member will reply here.` });
              break;
            case "error":
              toast.error("Something went wrong", { description: e.message });
              push({ kind: "notice", tone: "warning", text: "Sorry, something went wrong. Please try again." });
              break;
          }
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        toast.error("Couldn't reach SupportPilot", { description: message });
      } finally {
        // drop the placeholder if nothing was written (e.g. the turn ended with a ticket draft)
        setItems((xs) =>
          xs.flatMap((it) =>
            it.key !== aiKey || it.msg.kind !== "ai"
              ? [it]
              : it.msg.text
                ? [{ key: aiKey, msg: { ...it.msg, streaming: false, status: undefined } }]
                : [],
          ),
        );
        setBusy(false);
      }
    },
    [push, patchAi],
  );

  const send = useCallback(
    async (text: string) => {
      const files = uploads.done;
      if (!stored || busy || uploads.uploading || (!text.trim() && !files.length)) return;
      const message = text.trim() || (files.length === 1 ? "I've attached a file." : "I've attached some files.");
      const attachments = files.map((f) => ({ id: f.id!, filename: f.name, kind: f.kind ?? "", size_bytes: f.size }));
      const ids = attachments.map((a) => a.id);
      push({ kind: "customer", text: message, attachments });
      setInput("");
      uploads.clear();
      if (mode === "human") {
        // messages go to the support team; the stream only confirms storage
        for await (const e of api.chat(stored.sessionId, message, ids)) if (e.type === "error") toast.error(e.message);
        return;
      }
      await consume(api.chat(stored.sessionId, message, ids));
    },
    [stored, busy, mode, push, consume, uploads],
  );

  // send a question queued by openChat() once the session exists
  useEffect(() => {
    if (open && stored && queued.current && !busy) {
      const q = queued.current;
      queued.current = null;
      void send(q);
    }
  }, [open, stored, busy, send]);

  const decide = async (d: { action: "approve" | "edit" | "cancel"; subject?: string; description?: string }) => {
    if (!stored) return;
    setDraft(null);
    if (d.action === "cancel") push({ kind: "notice", tone: "info", text: "Ticket draft cancelled." });
    await consume(api.resume(stored.sessionId, d));
  };

  function resetSession() {
    localStorage.removeItem(STORAGE_KEY);
    setStored(null);
    setItems([]);
    setDraft(null);
    setMode("ai");
    lastStaffId.current = 0;
  }

  return (
    <>
      {!open && (
        <Button
          size="lg"
          onClick={() => setOpen(true)}
          className="fixed right-5 bottom-5 z-50 h-12 gap-2 rounded-full px-5 shadow-lg"
        >
          <MessageCircle className="size-5" /> Ask SupportPilot
        </Button>
      )}

      {open && (
        <section
          aria-label="SupportPilot chat"
          className="fixed right-5 bottom-5 z-50 flex h-[min(640px,calc(100vh-2.5rem))] w-[min(420px,calc(100vw-2.5rem))] flex-col overflow-hidden rounded-2xl border bg-background shadow-2xl"
          onDragOver={(e) => {
            if (!canAttach || !e.dataTransfer.types.includes("Files")) return;
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false);
          }}
          onDrop={(e) => {
            if (!canAttach) return;
            e.preventDefault();
            setDragging(false);
            uploads.add(e.dataTransfer.files);
          }}
        >
          {dragging && (
            <div className="pointer-events-none absolute inset-2 z-10 flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-primary bg-background/90 text-sm font-medium">
              <Paperclip className="size-6" /> Drop files to attach
              <span className="text-xs font-normal text-muted-foreground">Screenshots, PDFs or log files · up to 10 MB</span>
            </div>
          )}
          <header className="flex items-center gap-2 border-b px-4 py-3">
            <div className="flex size-8 items-center justify-center rounded-full bg-primary text-primary-foreground">
              {mode === "human" ? <Headset className="size-4" /> : <Sparkles className="size-4" />}
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold">SupportPilot</p>
              <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <span className={cn("size-1.5 rounded-full", mode === "human" ? "bg-emerald-500" : "bg-sky-500")} />
                {mode === "human" ? "Connected to the support team" : "AI assistant · answers from SearchUnify docs"}
              </p>
            </div>
            {stored && (
              <Button variant="ghost" size="icon" title="New conversation" onClick={resetSession}>
                <RotateCcw className="size-4" />
              </Button>
            )}
            <Button variant="ghost" size="icon" title="Close" onClick={() => setOpen(false)}>
              <X className="size-4" />
            </Button>
          </header>

          {!stored ? (
            <PreChatForm
              onStart={async (name, email) => {
                const s = await api.createSession(name, email);
                const next = { sessionId: s.session_id, name };
                localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
                setStored(next);
              }}
            />
          ) : (
            <>
              <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
                {items.length === 0 && !draft && (
                  <div className="space-y-3 py-6 text-center">
                    <p className="text-sm text-muted-foreground">
                      Hi {stored.name.split(" ")[0]}! Ask about content sources, search clients, analytics or anything
                      else in SearchUnify.
                    </p>
                    <div className="flex flex-col gap-1.5">
                      {["How do I add Jira as a content source?", "How can I see searches with no results?", "I need to talk to a person"].map(
                        (q) => (
                          <Button key={q} variant="outline" size="sm" className="justify-start" onClick={() => send(q)}>
                            {q}
                          </Button>
                        ),
                      )}
                    </div>
                  </div>
                )}
                {items.map((it) => (
                  <ChatMessage key={it.key} message={it.msg} />
                ))}
                {draft && <TicketDraftCard draft={draft} disabled={busy} onDecide={decide} />}
              </div>

              <form
                className="space-y-2 border-t p-3"
                onSubmit={(e) => {
                  e.preventDefault();
                  void send(input);
                }}
              >
                <PendingFileChips files={uploads.files} onRemove={uploads.remove} />
                <div className="flex items-end gap-2">
                  <AttachMenu onFiles={uploads.add} disabled={!canAttach} />
                  <Textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        void send(input);
                      }
                    }}
                    onPaste={(e) => {
                      // pasted screenshots arrive as files
                      const pasted = Array.from(e.clipboardData.files);
                      if (pasted.length && canAttach) {
                        e.preventDefault();
                        uploads.add(pasted);
                      }
                    }}
                    placeholder={
                      draft ? "Review the ticket draft above first" : mode === "human" ? "Message the support team…" : "Ask a question…"
                    }
                    disabled={busy || !!draft}
                    rows={1}
                    className="max-h-32 min-h-10 resize-none text-sm"
                  />
                  <Button
                    type="submit"
                    size="icon"
                    disabled={busy || !!draft || uploads.uploading || (!input.trim() && !uploads.done.length)}
                    aria-label="Send"
                  >
                    <SendHorizontal className="size-4" />
                  </Button>
                </div>
              </form>
            </>
          )}
        </section>
      )}
    </>
  );
}

function PreChatForm({ onStart }: { onStart: (name: string, email: string) => Promise<void> }) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [pending, setPending] = useState(false);
  return (
    <form
      className="flex flex-1 flex-col justify-center gap-4 p-6"
      onSubmit={async (e) => {
        e.preventDefault();
        setPending(true);
        try {
          await onStart(name.trim(), email.trim());
        } catch (err) {
          toast.error("Couldn't start the chat", { description: err instanceof Error ? err.message : String(err) });
        } finally {
          setPending(false);
        }
      }}
    >
      <div>
        <h2 className="text-lg font-semibold">Chat with SupportPilot</h2>
        <p className="text-sm text-muted-foreground">
          We use your email to look up and create support tickets for you.
        </p>
      </div>
      <div className="space-y-1.5">
        <Label htmlFor="chat-name">Name</Label>
        <Input id="chat-name" required value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" />
      </div>
      <div className="space-y-1.5">
        <Label htmlFor="chat-email">Work email</Label>
        <Input
          id="chat-email"
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          autoComplete="email"
        />
      </div>
      <Button type="submit" disabled={pending}>
        {pending ? "Starting…" : "Start chat"}
      </Button>
    </form>
  );
}
