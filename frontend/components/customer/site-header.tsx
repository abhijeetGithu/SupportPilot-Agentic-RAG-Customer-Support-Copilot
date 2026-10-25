import Link from "next/link";
import { Sparkles } from "lucide-react";

import { buttonVariants } from "@/components/ui/button";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-6 px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <span className="flex size-7 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Sparkles className="size-4" />
          </span>
          SupportPilot
          <span className="hidden text-sm font-normal text-muted-foreground sm:inline">for SearchUnify</span>
        </Link>
        <nav className="ml-auto flex items-center gap-1 text-sm">
          <Link href="/support/new" className={buttonVariants({ variant: "ghost", size: "sm" })}>
            Contact support
          </Link>
          <Link href="/support/tickets" className={buttonVariants({ variant: "ghost", size: "sm" })}>
            Track a ticket
          </Link>
          <Link href="/login" className={buttonVariants({ variant: "outline", size: "sm" })}>
            Staff login
          </Link>
        </nav>
      </div>
    </header>
  );
}
