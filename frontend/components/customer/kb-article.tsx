"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { BookOpenCheck } from "lucide-react";

import { openChat } from "@/components/chat/chat-widget";
import { ArticleBody } from "@/components/dashboard/knowledge-review";
import { formatDate } from "@/components/labels";
import { Button, buttonVariants } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, api } from "@/lib/api";

/** Published KB article: the page the assistant's "Support KB" citations link to. */
export function KbArticleView() {
  const { number } = useParams<{ number: string }>();
  const { data, error, isLoading } = useQuery({
    queryKey: ["public-kb", number],
    queryFn: () => api.publicKb(number),
    retry: false,
  });

  if (isLoading) return <Skeleton className="h-64 w-full rounded-xl" />;
  if (error || !data) {
    const missing = error instanceof ApiError && error.status === 404;
    return (
      <div className="rounded-xl border p-8 text-center">
        <p className="font-medium">{missing ? "This article isn't available." : "Couldn't load the article."}</p>
        <Link href="/" className={buttonVariants({ variant: "outline", className: "mt-4" })}>Back to support</Link>
      </div>
    );
  }
  return (
    <article className="space-y-6">
      <div>
        <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
          <BookOpenCheck className="size-4" /> Support knowledge base · {data.number}
          {data.product_area && <> · {data.product_area}</>}
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">{data.title}</h1>
        {data.published_at && (
          <p className="mt-1 text-sm text-muted-foreground">
            Written by our support team from a solved case · {formatDate(data.published_at)}
          </p>
        )}
      </div>
      <div className="rounded-xl border p-6">
        <ArticleBody text={data.body} />
      </div>
      <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
        Still stuck?
        <Button variant="outline" size="sm" onClick={() => openChat()}>Ask SupportPilot</Button>
        <Link href="/support/new" className={buttonVariants({ variant: "outline", size: "sm" })}>Contact support</Link>
      </div>
    </article>
  );
}
