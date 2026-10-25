import { Suspense } from "react";

import { TicketDetail } from "@/components/dashboard/ticket-detail";
import { Skeleton } from "@/components/ui/skeleton";

export default function TicketPage() {
  // the client component reads the [id] param, so it renders inside Suspense
  return (
    <Suspense fallback={<Skeleton className="h-96 w-full rounded-xl" />}>
      <TicketDetail />
    </Suspense>
  );
}
