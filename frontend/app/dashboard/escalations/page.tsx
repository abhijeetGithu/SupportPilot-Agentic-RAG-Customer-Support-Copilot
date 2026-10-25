import { EscalationQueue } from "@/components/dashboard/escalation-queue";

export default function EscalationsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Escalations</h1>
        <p className="text-sm text-muted-foreground">
          Customers who asked for a person, or whose issue the assistant handed over. Take one to reply in their chat.
        </p>
      </div>
      <EscalationQueue />
    </div>
  );
}
