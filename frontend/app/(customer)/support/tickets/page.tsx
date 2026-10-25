import { Suspense } from "react";

import { TicketTracker } from "@/components/customer/ticket-tracker";
import { Skeleton } from "@/components/ui/skeleton";

export default function TrackTicketPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-12">
      <h1 className="text-3xl font-semibold tracking-tight">Track a ticket</h1>
      <p className="mt-2 text-muted-foreground">Enter your ticket number and the email you used to open it.</p>
      <div className="mt-8">
        {/* reads ?number=&email= from the URL, so it renders on the client inside Suspense */}
        <Suspense fallback={<Skeleton className="h-24 w-full rounded-xl" />}>
          <TicketTracker />
        </Suspense>
      </div>
    </div>
  );
}
