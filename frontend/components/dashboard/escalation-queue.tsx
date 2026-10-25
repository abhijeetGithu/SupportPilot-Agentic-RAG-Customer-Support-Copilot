"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Siren } from "lucide-react";
import { toast } from "sonner";

import { PriorityBadge, formatDate } from "@/components/labels";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";

export function EscalationQueue() {
  const router = useRouter();
  const qc = useQueryClient();
  const { data, isPending } = useQuery({ queryKey: ["escalations"], queryFn: api.escalations, refetchInterval: 10_000 });
  const take = useMutation({
    mutationFn: (id: number) => api.takeEscalation(id),
    onSuccess: (d) => {
      void qc.invalidateQueries({ queryKey: ["escalations"] });
      toast.success(`You took ${d.ticket.number}`, { description: "Your replies appear live in the customer's chat." });
      router.push(`/dashboard/tickets/${d.ticket.id}`);
    },
    onError: (e) => toast.error("Couldn't take this ticket", { description: String(e) }),
  });

  if (isPending) return <Skeleton className="h-40 w-full rounded-xl" />;
  if (!data?.items.length)
    return (
      <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed py-16 text-center">
        <Siren className="size-6 text-muted-foreground" />
        <p className="font-medium">No customers waiting</p>
        <p className="text-sm text-muted-foreground">Escalations from the chat appear here. This page refreshes on its own.</p>
      </div>
    );

  return (
    <ul className="space-y-3">
      {data.items.map((t) => (
        <li key={t.id} className="flex flex-wrap items-center gap-4 rounded-xl border p-4">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs text-muted-foreground">{t.number}</span>
              <PriorityBadge priority={t.priority} />
              <span className="text-xs text-muted-foreground">waiting since {formatDate(t.updated_at)}</span>
            </div>
            <Link href={`/dashboard/tickets/${t.id}`} className="mt-1 block truncate font-medium hover:underline">
              {t.subject}
            </Link>
            <p className="truncate text-sm text-muted-foreground">
              {t.customer_name} · {t.customer_email} · {t.category}
            </p>
          </div>
          <Button disabled={take.isPending} onClick={() => take.mutate(t.id)}>
            Take conversation
          </Button>
        </li>
      ))}
    </ul>
  );
}
