import { Badge } from "@/components/ui/badge";
import type { Priority, TicketStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

const PRIORITY_STYLE: Record<Priority, string> = {
  urgent: "bg-red-600 text-white",
  high: "bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-200",
  medium: "bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-200",
  low: "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300",
};

const STATUS_STYLE: Record<TicketStatus, string> = {
  open: "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-200",
  in_progress: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-200",
  escalated: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-200",
  resolved: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200",
  closed: "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400",
};

export const STATUSES: TicketStatus[] = ["open", "in_progress", "escalated", "resolved", "closed"];
export const PRIORITIES: Priority[] = ["urgent", "high", "medium", "low"];
export const CATEGORIES = [
  "Billing and Payments",
  "Customer Service",
  "General Inquiry",
  "Human Resources",
  "IT Support",
  "Product Support",
  "Returns and Exchanges",
  "Sales and Pre-Sales",
  "Service Outages and Maintenance",
  "Technical Support",
];

export function PriorityBadge({ priority }: { priority: Priority }) {
  return <Badge className={cn("capitalize border-0", PRIORITY_STYLE[priority])}>{priority}</Badge>;
}

export function StatusBadge({ status }: { status: TicketStatus }) {
  return <Badge className={cn("border-0", STATUS_STYLE[status])}>{status.replace("_", " ")}</Badge>;
}

export function formatDate(iso: string) {
  // the API returns naive UTC timestamps
  const d = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** Native select styled like the shadcn inputs (accessible, no extra wiring). */
export function NativeSelect({ className, ...props }: React.ComponentProps<"select">) {
  return (
    <select
      className={cn(
        "h-9 rounded-md border border-input bg-background px-2.5 text-sm shadow-xs outline-none",
        "focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50",
        className,
      )}
      {...props}
    />
  );
}
