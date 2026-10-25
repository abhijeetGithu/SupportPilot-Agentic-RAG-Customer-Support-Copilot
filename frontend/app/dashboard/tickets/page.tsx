import { TicketInbox } from "@/components/dashboard/ticket-inbox";

export default function TicketsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Tickets</h1>
        <p className="text-sm text-muted-foreground">
          Urgent and high priority first. Tickets the classifier wasn’t sure about are marked for review.
        </p>
      </div>
      <TicketInbox />
    </div>
  );
}
