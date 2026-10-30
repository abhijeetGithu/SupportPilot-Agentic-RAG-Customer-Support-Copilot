"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Sparkles } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, staffToken } from "@/lib/api";

// public read-only account, e.g. NEXT_PUBLIC_DEMO_LOGIN="demo:look-around-only" (never the admin login)
const [demoUser, demoPassword] = (process.env.NEXT_PUBLIC_DEMO_LOGIN ?? "").split(":");
const DEMO = demoUser && demoPassword ? { user: demoUser, password: demoPassword } : null;

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [pending, setPending] = useState(false);

  return (
    <div className="flex flex-1 items-center justify-center p-4">
      <form
        className="w-full max-w-sm space-y-4 rounded-2xl border p-8 shadow-sm"
        onSubmit={async (e) => {
          e.preventDefault();
          setPending(true);
          try {
            const { access_token, role } = await api.login(username, password);
            staffToken.set(access_token, role);
            router.push("/dashboard");
          } catch {
            toast.error("Wrong username or password");
          } finally {
            setPending(false);
          }
        }}
      >
        <div className="flex items-center gap-2">
          <span className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Sparkles className="size-4" />
          </span>
          <div>
            <h1 className="font-semibold leading-tight">SupportPilot</h1>
            <p className="text-xs text-muted-foreground">Staff dashboard</p>
          </div>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="username">Username</Label>
          <Input id="username" required value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="password">Password</Label>
          <Input
            id="password"
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
          />
        </div>
        <Button type="submit" className="w-full" disabled={pending}>
          {pending ? "Signing in…" : "Sign in"}
        </Button>
        {DEMO ? (
          <div className="space-y-2 rounded-lg bg-muted/60 p-3 text-xs">
            <p>
              <span className="font-medium">Exploring the project?</span> Use the read-only demo login:{" "}
              <span className="font-mono">{DEMO.user} / {DEMO.password}</span>
            </p>
            <Button type="button" size="sm" variant="outline" className="w-full"
                    onClick={() => { setUsername(DEMO.user); setPassword(DEMO.password); }}>
              Fill in the demo login
            </Button>
          </div>
        ) : (
          <p className="text-center text-xs text-muted-foreground">Local development: admin / admin</p>
        )}
        <p className="text-center text-xs text-muted-foreground">
          <Link href="/" className="underline">
            Back to the site
          </Link>
        </p>
      </form>
    </div>
  );
}
