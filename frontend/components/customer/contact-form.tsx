"use client";

import Link from "next/link";
import { useRef, useState } from "react";
import { CheckCircle2, Upload } from "lucide-react";
import { toast } from "sonner";

import { ACCEPT, MAX_FILES, PendingFileChips, useUploads } from "@/components/attachments";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/api";
import type { Ticket } from "@/lib/types";
import { cn } from "@/lib/utils";

export function ContactForm() {
  const [form, setForm] = useState({ customer_name: "", customer_email: "", subject: "", description: "" });
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<{ ticket: Ticket; duplicate: boolean; files: number } | null>(null);
  const uploads = useUploads();
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));

  if (result) {
    const { ticket, duplicate } = result;
    return (
      <div className="rounded-xl border p-6">
        <CheckCircle2 className="mb-3 size-8 text-emerald-600" />
        <h2 className="text-xl font-semibold">
          {duplicate ? `Added to your open ticket ${ticket.number}` : `Ticket ${ticket.number} created`}
        </h2>
        <p className="mt-2 text-sm text-muted-foreground">
          {duplicate
            ? "This looks like the same issue as a ticket you opened recently, so we added your details to it."
            : `It's been routed to ${ticket.category ?? "our team"}. We'll reply by email and on your ticket page.`}
        </p>
        {result.files > 0 && (
          <p className="mt-2 text-sm text-muted-foreground">
            {result.files} file{result.files === 1 ? "" : "s"} attached for the support team.
          </p>
        )}
        <p className="mt-4 text-sm">
          Keep your ticket number <span className="font-mono font-semibold">{ticket.number}</span> to track it.
        </p>
        <div className="mt-6 flex gap-2">
          <Link
            href={`/support/tickets?number=${ticket.number}&email=${encodeURIComponent(ticket.customer_email)}`}
            className={buttonVariants()}
          >
            View ticket
          </Link>
          <Button variant="outline" onClick={() => setResult(null)}>
            Open another
          </Button>
        </div>
      </div>
    );
  }

  return (
    <form
      className="space-y-4 rounded-xl border p-6"
      onSubmit={async (e) => {
        e.preventDefault();
        setPending(true);
        try {
          const r = await api.createTicket({ ...form, attachment_ids: uploads.ids });
          setResult({ ticket: r.ticket, duplicate: r.duplicate_of_existing, files: uploads.ids.length });
          uploads.clear();
        } catch (err) {
          toast.error("Couldn't create the ticket", { description: err instanceof Error ? err.message : String(err) });
        } finally {
          setPending(false);
        }
      }}
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor="name">Name</Label>
          <Input id="name" required value={form.customer_name} onChange={set("customer_name")} autoComplete="name" />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="email">Work email</Label>
          <Input id="email" type="email" required value={form.customer_email} onChange={set("customer_email")} autoComplete="email" />
        </div>
      </div>
      <div className="space-y-1.5">
        <Label htmlFor="subject">Subject</Label>
        <Input id="subject" required minLength={3} maxLength={255} value={form.subject} onChange={set("subject")} />
      </div>
      <div className="space-y-1.5">
        <Label htmlFor="description">What’s happening?</Label>
        <Textarea
          id="description"
          required
          minLength={10}
          rows={6}
          value={form.description}
          onChange={set("description")}
          placeholder="What you were trying to do, what happened, any error messages, and which content source or search client is affected."
        />
      </div>
      <div className="space-y-2">
        <Label>
          Attachments <span className="font-normal text-muted-foreground">(optional)</span>
        </Label>
        <input
          ref={fileInput}
          type="file"
          multiple
          accept={ACCEPT}
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.length) uploads.add(e.target.files);
            e.target.value = "";
          }}
        />
        <button
          type="button"
          onClick={() => fileInput.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            uploads.add(e.dataTransfer.files);
          }}
          className={cn(
            "flex w-full flex-col items-center gap-1 rounded-lg border-2 border-dashed px-4 py-5 text-sm transition-colors",
            dragging ? "border-primary bg-primary/5" : "hover:bg-muted/50",
          )}
        >
          <Upload className="size-5 text-muted-foreground" />
          <span>
            <span className="font-medium">Drop files here</span> or click to browse
          </span>
          <span className="text-xs text-muted-foreground">
            Screenshots, PDFs or log files · up to {MAX_FILES} files, 10 MB each
          </span>
        </button>
        <PendingFileChips files={uploads.files} onRemove={uploads.remove} />
      </div>
      <p className="text-xs text-muted-foreground">
        Your ticket is categorised and prioritised automatically. Don’t include passwords or card numbers; they are
        removed before saving.
      </p>
      <Button type="submit" disabled={pending || uploads.uploading}>
        {pending ? "Submitting…" : uploads.uploading ? "Uploading files…" : "Submit ticket"}
      </Button>
    </form>
  );
}
