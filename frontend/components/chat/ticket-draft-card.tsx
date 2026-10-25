"use client";

import { useState } from "react";
import { Ticket as TicketIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type { TicketDraft } from "@/lib/types";

type Decision = { action: "approve" | "edit" | "cancel"; subject?: string; description?: string };

/** Human-in-the-loop: the agent paused; the customer approves, edits or cancels the ticket draft. */
export function TicketDraftCard({
  draft,
  disabled,
  onDecide,
}: {
  draft: TicketDraft;
  disabled: boolean;
  onDecide: (d: Decision) => void;
}) {
  const [subject, setSubject] = useState(draft.subject);
  const [description, setDescription] = useState(draft.description);
  const edited = subject !== draft.subject || description !== draft.description;

  return (
    <div className="rounded-xl border-2 border-dashed border-primary/30 bg-background p-3 shadow-sm">
      <p className="mb-2 flex items-center gap-1.5 text-sm font-medium">
        <TicketIcon className="size-4" /> Review your support ticket
      </p>
      <div className="space-y-2">
        <div className="space-y-1">
          <Label htmlFor="draft-subject" className="text-xs text-muted-foreground">
            Subject
          </Label>
          <Input id="draft-subject" value={subject} onChange={(e) => setSubject(e.target.value)} disabled={disabled} />
        </div>
        <div className="space-y-1">
          <Label htmlFor="draft-description" className="text-xs text-muted-foreground">
            Description
          </Label>
          <Textarea
            id="draft-description"
            rows={4}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            disabled={disabled}
            className="text-sm"
          />
        </div>
      </div>
      <div className="mt-3 flex justify-end gap-2">
        <Button variant="ghost" size="sm" disabled={disabled} onClick={() => onDecide({ action: "cancel" })}>
          Cancel
        </Button>
        <Button
          size="sm"
          disabled={disabled || subject.trim().length < 3 || description.trim().length < 10}
          onClick={() => onDecide(edited ? { action: "edit", subject, description } : { action: "approve" })}
        >
          Create ticket
        </Button>
      </div>
    </div>
  );
}
