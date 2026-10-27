import { Suspense } from "react";

import { KbArticleView } from "@/components/customer/kb-article";
import { Skeleton } from "@/components/ui/skeleton";

export default function KbArticlePage() {
  // the client component reads the [number] param, so it renders inside Suspense
  return (
    <div className="mx-auto max-w-3xl px-4 py-12">
      <Suspense fallback={<Skeleton className="h-64 w-full rounded-xl" />}>
        <KbArticleView />
      </Suspense>
    </div>
  );
}
