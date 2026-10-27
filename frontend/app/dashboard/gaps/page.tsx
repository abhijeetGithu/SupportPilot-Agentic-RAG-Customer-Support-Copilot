import { GapReportView } from "@/components/dashboard/gap-report";

export default function GapsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Doc gaps</h1>
        <p className="text-sm text-muted-foreground">
          What customers ask that the documentation doesn&apos;t cover well, from every chat question&apos;s best
          doc match and outcome.
        </p>
      </div>
      <GapReportView />
    </div>
  );
}
