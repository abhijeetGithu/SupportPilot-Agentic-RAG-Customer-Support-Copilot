import { IncidentList } from "@/components/dashboard/incident-list";

export default function IncidentsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Incidents</h1>
        <p className="text-sm text-muted-foreground">
          Detected automatically when several customers report the same problem within two hours. Post one notice,
          update everyone at once, and resolve their tickets together.
        </p>
      </div>
      <IncidentList />
    </div>
  );
}
