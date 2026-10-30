"use client";

import { useSyncExternalStore } from "react";

import { staffToken } from "@/lib/api";

const subscribe = (onChange: () => void) => {
  window.addEventListener("storage", onChange); // sign-in / sign-out in another tab
  return () => window.removeEventListener("storage", onChange);
};
const noop = () => () => {};

/**
 * Reads the staff token from localStorage without setState-in-effect and without hydration
 * mismatches: the server snapshot (and the hydration render) see `mounted: false`.
 */
export function useStaffToken() {
  const mounted = useSyncExternalStore(noop, () => true, () => false);
  const token = useSyncExternalStore(subscribe, () => staffToken.get(), () => null);
  const role = useSyncExternalStore(subscribe, () => staffToken.role(), () => null);
  return { mounted, token, role };
}
