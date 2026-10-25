"use client";

import { ArrowRight, MessageCircle } from "lucide-react";

import { openChat } from "@/components/chat/chat-widget";
import { Button } from "@/components/ui/button";

export function OpenChatButton() {
  return (
    <Button size="lg" className="gap-2" onClick={() => openChat()}>
      <MessageCircle className="size-4" /> Chat with SupportPilot
    </Button>
  );
}

export function SuggestedQuestions({ questions }: { questions: string[] }) {
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      {questions.map((q) => (
        <button
          key={q}
          onClick={() => openChat(q)}
          className="group flex items-center justify-between gap-3 rounded-lg border bg-background px-4 py-3 text-left text-sm transition-colors hover:bg-muted"
        >
          {q}
          <ArrowRight className="size-4 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
        </button>
      ))}
    </div>
  );
}
