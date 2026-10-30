"use client";

import { useEffect, useState } from "react";
import { Loader2, TriangleAlert } from "lucide-react";

import { api } from "@/lib/api";

/** The demo backend runs on a free tier that sleeps when idle: say so while it wakes up, instead of
 *  letting the first request look broken. Invisible when the backend answers quickly. */
export function BackendStatus() {
  const [state, setState] = useState<"checking" | "waking" | "ok" | "down">("checking");

  useEffect(() => {
    let cancelled = false;
    const started = Date.now();
    const slow = setTimeout(() => setState((s) => (s === "checking" ? "waking" : s)), 2500);
    (async () => {
      while (!cancelled && Date.now() - started < 180_000) {
        try {
          await api.health();
          if (!cancelled) setState("ok");
          return;
        } catch {
          await new Promise((r) => setTimeout(r, 4000));
        }
      }
      if (!cancelled) setState("down");
    })();
    return () => {
      cancelled = true;
      clearTimeout(slow);
    };
  }, []);

  if (state === "waking")
    return (
      <div role="status" className="flex items-center justify-center gap-2 bg-amber-50 px-4 py-2 text-sm text-amber-900 dark:bg-amber-950 dark:text-amber-100">
        <Loader2 className="size-4 animate-spin" />
        Waking up the AI backend. It runs on a free tier that sleeps when idle, so this takes about a minute.
      </div>
    );
  if (state === "down")
    return (
      <div role="alert" className="flex items-center justify-center gap-2 bg-red-50 px-4 py-2 text-sm text-red-900 dark:bg-red-950 dark:text-red-100">
        <TriangleAlert className="size-4" /> The AI backend isn&apos;t responding right now. Please try again later.
      </div>
    );
  return null;
}
