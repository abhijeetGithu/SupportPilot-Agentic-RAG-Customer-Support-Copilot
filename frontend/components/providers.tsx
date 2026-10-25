"use client";

import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { Toaster } from "@/components/ui/sonner";

let browserQueryClient: QueryClient | undefined;

function getQueryClient() {
  const make = () => new QueryClient({ defaultOptions: { queries: { staleTime: 5_000, retry: 1 } } });
  // Keep server renders isolated; reuse one client in the browser.
  if (typeof window === "undefined") return make();
  browserQueryClient ??= make();
  return browserQueryClient;
}

export function Providers({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider client={getQueryClient()}>
      {children}
      <Toaster richColors position="top-right" />
    </QueryClientProvider>
  );
}
