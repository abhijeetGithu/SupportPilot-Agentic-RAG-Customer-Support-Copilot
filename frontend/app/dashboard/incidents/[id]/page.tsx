import { Suspense } from "react";

import { IncidentDetailView } from "@/components/dashboard/incident-detail";
import { Skeleton } from "@/components/ui/skeleton";

export default function IncidentPage() {
  // the client component reads the [id] param, so it renders inside Suspense
  return (
    <Suspense fallback={<Skeleton className="h-96 w-full rounded-xl" />}>
      <IncidentDetailView />
    </Suspense>
  );
}
