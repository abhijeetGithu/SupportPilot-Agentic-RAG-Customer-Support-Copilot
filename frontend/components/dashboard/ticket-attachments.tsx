"use client";

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { CircleAlert, Download, FileText, Paperclip, Sparkles, TriangleAlert } from "lucide-react";
import { toast } from "sonner";

import { formatSize } from "@/components/attachments";
import { formatDate } from "@/components/labels";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import type { Attachment, FileAnalysis } from "@/lib/types";

async function download(att: Attachment) {
  try {
    const url = URL.createObjectURL(await api.attachmentBlob(att.id));
    const a = document.createElement("a");
    a.href = url;
    a.download = att.filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) {
    toast.error("Download failed", { description: String(e) });
  }
}

/** Image preview fetched with the staff token (attachments are never public URLs). */
function Thumbnail({ att }: { att: Attachment }) {
  const { data: url } = useQuery({
    queryKey: ["attachment-preview", att.id],
    queryFn: async () => URL.createObjectURL(await api.attachmentBlob(att.id)),
    staleTime: Infinity,
    gcTime: 0, // drop the cached object URL when unmounted, it is revoked below
  });
  useEffect(() => () => {
    if (url) URL.revokeObjectURL(url);
  }, [url]);
  if (!url) return <Skeleton className="h-28 w-full rounded-md" />;
  return (
    <a href={url} target="_blank" rel="noreferrer" title="Open full size">
      {/* eslint-disable-next-line @next/next/no-img-element -- blob URL, not optimisable by next/image */}
      <img src={url} alt={att.filename} className="h-28 w-full rounded-md border object-cover object-top" />
    </a>
  );
}

const SOURCE_LABEL: Record<FileAnalysis["source"], string> = {
  vision: "Read by AI",
  pdf_text: "PDF text",
  pdf_vision: "Scanned PDF, read by AI",
};

/** What the vision model / PDF reader found, so staff can triage without opening the file. */
function AiReading({ analysis }: { analysis: FileAnalysis }) {
  return (
    <div className="space-y-1.5 border-t pt-1.5 text-[11px]">
      <p className="flex min-w-0 items-center gap-1 font-medium text-muted-foreground">
        <Sparkles className="size-3 shrink-0" />
        <span className="shrink-0 whitespace-nowrap">{SOURCE_LABEL[analysis.source]}</span>
        {analysis.product_area && (
          <span className="min-w-0 truncate font-normal" title={analysis.product_area}>
            · {analysis.product_area}
          </span>
        )}
      </p>
      {analysis.summary && <p className="leading-snug">{analysis.summary}</p>}
      {analysis.error_messages.length > 0 && (
        <ul className="space-y-1">
          {analysis.error_messages.map((e) => (
            <li key={e} className="flex gap-1 rounded bg-destructive/5 px-1.5 py-1 font-mono text-destructive">
              <CircleAlert className="mt-px size-3 shrink-0" />
              <span className="break-words">{e}</span>
            </li>
          ))}
        </ul>
      )}
      {analysis.contains_instructions && (
        <p className="flex gap-1 rounded bg-amber-500/10 px-1.5 py-1 text-amber-800 dark:text-amber-300">
          <TriangleAlert className="mt-px size-3 shrink-0" />
          Contains text addressed to an AI (possible prompt injection). The assistant treated it as data.
        </p>
      )}
      {analysis.visible_text && (
        <details>
          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">Visible text</summary>
          <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-muted p-1.5 font-mono">
            {analysis.visible_text}
          </pre>
        </details>
      )}
    </div>
  );
}

export function TicketAttachments({ attachments }: { attachments: Attachment[] }) {
  if (!attachments.length) return null;
  return (
    <div className="border-b p-4">
      <p className="mb-2 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        <Paperclip className="size-3.5" /> Attachments ({attachments.length})
      </p>
      <ul className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {attachments.map((a) => (
          <li key={a.id} className="space-y-1.5 rounded-lg border p-2">
            {a.kind === "image" ? (
              <Thumbnail att={a} />
            ) : (
              <div className="flex h-28 items-center justify-center rounded-md bg-muted">
                <FileText className="size-8 text-muted-foreground" />
              </div>
            )}
            <div className="flex items-center gap-2">
              <div className="min-w-0 flex-1">
                <p className="truncate text-xs font-medium" title={a.filename}>{a.filename}</p>
                <p className="text-[11px] text-muted-foreground">
                  {formatSize(a.size_bytes)} · {formatDate(a.created_at)}
                </p>
              </div>
              <Button variant="ghost" size="icon-sm" aria-label={`Download ${a.filename}`} onClick={() => download(a)}>
                <Download className="size-4" />
              </Button>
            </div>
            {a.analysis && <AiReading analysis={a.analysis} />}
          </li>
        ))}
      </ul>
    </div>
  );
}
