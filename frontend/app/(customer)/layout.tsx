import { BackendStatus } from "@/components/backend-status";
import { ChatWidget } from "@/components/chat/chat-widget";
import { SiteHeader } from "@/components/customer/site-header";

export default function CustomerLayout({ children }: LayoutProps<"/">) {
  return (
    <>
      <BackendStatus />
      <SiteHeader />
      <main className="flex-1">{children}</main>
      <footer className="border-t py-6 text-center text-xs text-muted-foreground">
        SupportPilot is a portfolio project. Answers come from the public SearchUnify documentation, used with permission.
      </footer>
      <ChatWidget />
    </>
  );
}
