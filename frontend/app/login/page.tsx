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
            const { access_token } = await api.login(username, password);
            staffToken.set(access_token);
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
        <p className="text-center text-xs text-muted-foreground">
          Demo account: admin / admin ·{" "}
          <Link href="/" className="underline">
            back to the site
          </Link>
        </p>
      </form>
    </div>
  );
}
