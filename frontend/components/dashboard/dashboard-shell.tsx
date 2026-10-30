"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Suspense, useEffect, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity, BarChart3, BookOpenCheck, Inbox, LogOut, SearchX, Siren, Sparkles } from "lucide-react";

import { BackendStatus } from "@/components/backend-status";
import { Button } from "@/components/ui/button";
import { api, staffToken } from "@/lib/api";
import { useStaffToken } from "@/lib/use-staff-token";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/dashboard", label: "Overview", icon: BarChart3 },
  { href: "/dashboard/tickets", label: "Tickets", icon: Inbox },
  { href: "/dashboard/escalations", label: "Escalations", icon: Siren },
  { href: "/dashboard/incidents", label: "Incidents", icon: Activity },
  { href: "/dashboard/knowledge", label: "Knowledge", icon: BookOpenCheck },
  { href: "/dashboard/gaps", label: "Doc gaps", icon: SearchX },
];

/** Client-side auth guard + sidebar. The API enforces auth; this only redirects to /login. */
export function DashboardShell({ children }: { children: ReactNode }) {
  const router = useRouter();
  const { mounted, token, role } = useStaffToken();
  const ready = mounted && !!token;

  useEffect(() => {
    if (mounted && !token) router.replace("/login");
  }, [mounted, token, router]);

  const escalations = useQuery({
    queryKey: ["escalations"],
    queryFn: api.escalations,
    enabled: ready,
    refetchInterval: 10_000,
  });
  const activeIncidents = useQuery({
    queryKey: ["incidents", "active"],
    queryFn: () => api.incidents("active"),
    enabled: ready,
    refetchInterval: 15_000,
  });
  const counts = {
    "/dashboard/escalations": escalations.data?.total,
    "/dashboard/incidents": activeIncidents.data?.length,
  };

  useEffect(() => {
    // expired / invalid token: the API client cleared it on 401
    if (escalations.error && !staffToken.get()) router.replace("/login");
  }, [escalations.error, router]);

  if (!ready) return null;

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 flex h-screen w-56 shrink-0 flex-col border-r bg-muted/30 p-3">
        <Link href="/" className="mb-6 flex items-center gap-2 px-2 pt-1 font-semibold">
          <span className="flex size-7 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Sparkles className="size-4" />
          </span>
          SupportPilot
        </Link>
        {/* the active link depends on the URL, which on dynamic routes is only known at runtime */}
        <Suspense fallback={<SidebarNav pathname={null} counts={counts} />}>
          <ActiveSidebarNav counts={counts} />
        </Suspense>
        {role === "viewer" && (
          <p className="mt-auto mb-2 rounded-md bg-amber-100 px-2.5 py-2 text-xs text-amber-900 dark:bg-amber-950 dark:text-amber-200">
            Read-only demo account: you can explore everything, but changes are disabled.
          </p>
        )}
        <Button
          variant="ghost"
          size="sm"
          className={cn("justify-start gap-2 text-muted-foreground", role !== "viewer" && "mt-auto")}
          onClick={() => {
            staffToken.clear();
            router.push("/login");
          }}
        >
          <LogOut className="size-4" /> Sign out
        </Button>
      </aside>
      <main className="min-w-0 flex-1">
        <BackendStatus />
        <div className="p-6 lg:p-8">{children}</div>
      </main>
    </div>
  );
}

type Counts = Record<string, number | undefined>;

function ActiveSidebarNav({ counts }: { counts: Counts }) {
  return <SidebarNav pathname={usePathname()} counts={counts} />;
}

function SidebarNav({ pathname, counts }: { pathname: string | null; counts: Counts }) {
  return (
    <nav className="flex flex-col gap-1">
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = pathname !== null && (href === "/dashboard" ? pathname === href : pathname.startsWith(href));
        const count = counts[href];
        return (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex items-center gap-2 rounded-md px-2.5 py-2 text-sm transition-colors",
              active ? "bg-background font-medium shadow-sm" : "text-muted-foreground hover:bg-background/60",
            )}
          >
            <Icon className="size-4" />
            <span className="flex-1">{label}</span>
            {!!count && <span className="rounded-full bg-red-600 px-1.5 text-[11px] font-semibold text-white">{count}</span>}
          </Link>
        );
      })}
    </nav>
  );
}
