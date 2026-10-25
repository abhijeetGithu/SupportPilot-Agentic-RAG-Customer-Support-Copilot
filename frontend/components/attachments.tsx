"use client";

import { useCallback, useRef, useState } from "react";
import { FileImage, FileText, FileType2, Files, Loader2, Paperclip, ScrollText, TriangleAlert, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { api } from "@/lib/api";
import type { AttachmentMeta } from "@/lib/types";
import { cn } from "@/lib/utils";

// Mirrors the server allowlist (src/supportpilot/attachments.py); the server re-checks everything.
const IMAGE_TYPES = ".png,.jpg,.jpeg,.gif,.webp";
const DOC_TYPES = ".pdf";
const TEXT_TYPES = ".txt,.log,.csv,.json,.md";
export const ACCEPT = [IMAGE_TYPES, DOC_TYPES, TEXT_TYPES].join(",");
const ALLOWED = new Set(ACCEPT.split(","));
export const MAX_FILES = 5;
const MAX_MB = 10;

export type PendingFile = {
  key: string;
  name: string;
  size: number;
  status: "uploading" | "done" | "error";
  id?: string;
  kind?: string;
  error?: string;
};

export function formatSize(bytes: number) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const extOf = (name: string) => name.slice(name.lastIndexOf(".")).toLowerCase();
let seq = 0;

/** Uploads files as soon as they are picked; the form/message sends only the resulting ids. */
export function useUploads(sessionId?: string) {
  const [files, setFiles] = useState<PendingFile[]>([]);

  const add = useCallback(
    (list: FileList | File[]) => {
      // Side effects (uploads) stay outside state updaters, which React may run twice.
      let room = MAX_FILES - files.filter((f) => f.status !== "error").length;
      const entries: PendingFile[] = [];
      const toUpload: { key: string; file: File }[] = [];
      for (const file of Array.from(list)) {
        const key = `f${++seq}`;
        const name = file.name || `pasted-image-${seq}.png`;
        let error: string | undefined;
        if (room <= 0) error = `At most ${MAX_FILES} files`;
        else if (!ALLOWED.has(extOf(name))) error = "Unsupported file type";
        else if (file.size > MAX_MB * 1024 * 1024) error = `Larger than ${MAX_MB} MB`;
        entries.push({
          key,
          name,
          size: file.size,
          status: error ? "error" : "uploading",
          error,
        });
        if (!error) {
          room -= 1;
          toUpload.push({
            key,
            file: file.name ? file : new File([file], name, { type: file.type }),
          });
        }
      }
      setFiles((fs) => [...fs, ...entries]);
      for (const { key, file } of toUpload) {
        api
          .upload(file, sessionId)
          .then((a) =>
            setFiles((fs) => fs.map((f) => (f.key === key ? { ...f, status: "done", id: a.id, kind: a.kind } : f))),
          )
          .catch((e: Error) =>
            setFiles((fs) => fs.map((f) => (f.key === key ? { ...f, status: "error", error: e.message } : f))),
          );
      }
    },
    [files, sessionId],
  );

  const remove = useCallback((key: string) => setFiles((fs) => fs.filter((f) => f.key !== key)), []);
  const clear = useCallback(() => setFiles([]), []);
  const uploading = files.some((f) => f.status === "uploading");
  const done = files.filter((f) => f.status === "done");

  return {
    files,
    add,
    remove,
    clear,
    uploading,
    done,
    ids: done.map((f) => f.id!),
  };
}

function FileIcon({ kind, name }: { kind?: string; name: string }) {
  const image = kind === "image" || /\.(png|jpe?g|gif|webp)$/i.test(name);
  return image ? <FileImage className="size-3.5 shrink-0" /> : <FileText className="size-3.5 shrink-0" />;
}

/** Chips for files being attached (with remove buttons and upload state). */
export function PendingFileChips({ files, onRemove }: { files: PendingFile[]; onRemove: (key: string) => void }) {
  if (!files.length) return null;
  return (
    <ul className="flex flex-wrap gap-1.5">
      {files.map((f) => (
        <li
          key={f.key}
          title={f.error ?? f.name}
          className={cn(
            "flex max-w-full items-center gap-1.5 rounded-md border px-2 py-1 text-xs",
            f.status === "error" ? "border-destructive/40 bg-destructive/5 text-destructive" : "bg-muted/50",
          )}
        >
          {f.status === "uploading" ? (
            <Loader2 className="size-3.5 shrink-0 animate-spin" />
          ) : f.status === "error" ? (
            <TriangleAlert className="size-3.5 shrink-0" />
          ) : (
            <FileIcon kind={f.kind} name={f.name} />
          )}
          <span className="max-w-40 truncate">{f.name}</span>
          <span className="shrink-0 text-muted-foreground">{f.error ?? formatSize(f.size)}</span>
          <button
            type="button"
            onClick={() => onRemove(f.key)}
            aria-label={`Remove ${f.name}`}
            className="rounded hover:bg-background"
          >
            <X className="size-3" />
          </button>
        </li>
      ))}
    </ul>
  );
}

/** Read-only chips for files already sent (chat bubbles, ticket pages). */
export function SentFileChips({ files, className }: { files: AttachmentMeta[]; className?: string }) {
  if (!files.length) return null;
  return (
    <ul className={cn("flex flex-wrap gap-1.5", className)}>
      {files.map((f) => (
        <li
          key={f.id}
          className="flex items-center gap-1 rounded-md border bg-background/80 px-1.5 py-0.5 text-[11px] text-foreground"
        >
          <Paperclip className="size-3 shrink-0" />
          <span className="max-w-40 truncate">{f.filename}</span>
          <span className="text-muted-foreground">{formatSize(f.size_bytes)}</span>
        </li>
      ))}
    </ul>
  );
}

const PICKS = [
  {
    label: "Image or screenshot",
    hint: "PNG, JPG, GIF, WebP",
    accept: IMAGE_TYPES,
    Icon: FileImage,
  },
  { label: "PDF document", hint: "PDF", accept: DOC_TYPES, Icon: FileType2 },
  {
    label: "Log or text file",
    hint: "LOG, TXT, CSV, JSON, MD · the assistant can read these",
    accept: TEXT_TYPES,
    Icon: ScrollText,
  },
] as const;

/** Paperclip button that opens a menu of file types; each option opens the file picker filtered to that type. */
export function AttachMenu({ onFiles, disabled }: { onFiles: (files: FileList) => void; disabled?: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  const pick = (accept: string) => {
    if (!input.current) return;
    input.current.accept = accept;
    input.current.click();
  };
  return (
    <>
      <input
        ref={input}
        type="file"
        multiple
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.length) onFiles(e.target.files);
          e.target.value = ""; // allow picking the same file again
        }}
      />
      <DropdownMenu>
        <DropdownMenuTrigger
          disabled={disabled}
          render={<Button type="button" variant="ghost" size="icon" aria-label="Attach files" title="Attach files" />}
        >
          <Paperclip className="size-4" />
        </DropdownMenuTrigger>
        <DropdownMenuContent side="top" align="start" className="w-72">
          <DropdownMenuGroup>
            <DropdownMenuLabel>
              Attach a file · up to {MAX_FILES}, {MAX_MB} MB each
            </DropdownMenuLabel>
            {PICKS.map(({ label, hint, accept, Icon }) => (
              <DropdownMenuItem key={label} onClick={() => pick(accept)} className="items-start gap-2.5 py-2">
                <Icon className="mt-0.5 size-4 shrink-0" />
                <span className="flex flex-col">
                  <span>{label}</span>
                  <span className="text-xs text-muted-foreground">{hint}</span>
                </span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuGroup>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={() => pick(ACCEPT)} className="gap-2.5">
            <Files className="size-4" /> Any supported file
          </DropdownMenuItem>
          <p className="px-2 pt-1 pb-1.5 text-xs text-muted-foreground">
            Tip: paste a screenshot with Ctrl+V or drag files here.
          </p>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  );
}
