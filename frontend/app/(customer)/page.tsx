import Link from "next/link";
import { BookOpenText, LifeBuoy, ShieldCheck, Ticket } from "lucide-react";

import { OpenChatButton, SuggestedQuestions } from "@/components/customer/ask-buttons";
import { buttonVariants } from "@/components/ui/button";

const FEATURES = [
  {
    icon: BookOpenText,
    title: "Answers from the docs",
    text: "Hybrid search over 1,500+ passages of the SearchUnify documentation, with a source link for every step.",
  },
  {
    icon: Ticket,
    title: "Tickets you approve",
    text: "When the docs can't solve it, the assistant drafts a ticket. You review and edit it before it's submitted.",
  },
  {
    icon: LifeBuoy,
    title: "A person when you need one",
    text: "Ask for a human at any time. Your conversation is handed to the support team in the same chat.",
  },
  {
    icon: ShieldCheck,
    title: "Your data stays yours",
    text: "Tickets are tied to your email. The assistant can't open or reveal anyone else's tickets.",
  },
];

export default function Home() {
  return (
    <div className="mx-auto max-w-6xl px-4">
      <section className="py-16 md:py-24">
        <p className="mb-3 text-sm font-medium text-muted-foreground">SearchUnify Support</p>
        <h1 className="max-w-3xl text-4xl font-semibold tracking-tight md:text-5xl">
          Get answers about SearchUnify in seconds, and a real person when you need one.
        </h1>
        <p className="mt-4 max-w-2xl text-lg text-muted-foreground">
          SupportPilot answers from the official documentation, creates and tracks support tickets for you, and hands
          over to the support team when needed.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <OpenChatButton />
          <Link href="/support/new" className={buttonVariants({ variant: "outline", size: "lg" })}>
            Contact support
          </Link>
        </div>
      </section>

      <section className="pb-16">
        <h2 className="mb-4 text-sm font-medium text-muted-foreground">Popular questions</h2>
        <SuggestedQuestions
          questions={[
            "How do I set up Jira as a content source?",
            "How can I see which searches returned no results?",
            "How long is a refresh token valid for the Search API?",
            "How do I clone a search client?",
          ]}
        />
      </section>

      <section className="grid gap-4 pb-20 sm:grid-cols-2 lg:grid-cols-4">
        {FEATURES.map(({ icon: Icon, title, text }) => (
          <div key={title} className="rounded-xl border p-5">
            <Icon className="mb-3 size-5" />
            <h3 className="mb-1 font-medium">{title}</h3>
            <p className="text-sm text-muted-foreground">{text}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
