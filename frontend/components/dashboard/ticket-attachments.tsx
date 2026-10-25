"use client";

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Download, FileText, Paperclip } from "lucide-react";
import { toast } from "sonner";

import { formatSize } from "@/components/attachments";
import { formatDate } from "@/components/labels";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import type { Attachment } from "@/lib/types";

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
          </li>
        ))}
      </ul>
    </div>
  );
}
