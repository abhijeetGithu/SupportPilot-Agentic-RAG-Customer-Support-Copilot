import { KnowledgeReview } from "@/components/dashboard/knowledge-review";

export default function KnowledgePage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Knowledge</h1>
        <p className="text-sm text-muted-foreground">
          Articles the AI drafted from resolved tickets. Review and publish them, and the assistant starts answering
          those questions itself, citing the article.
        </p>
      </div>
      <KnowledgeReview />
    </div>
  );
}
