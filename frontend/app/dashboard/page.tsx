import { MetricsOverview } from "@/components/dashboard/metrics-overview";

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
        <p className="text-sm text-muted-foreground">Live ticket volume and how much the assistant resolves on its own.</p>
      </div>
      <MetricsOverview />
    </div>
  );
}
