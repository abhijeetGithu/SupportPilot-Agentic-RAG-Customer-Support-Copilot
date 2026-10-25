import { ContactForm } from "@/components/customer/contact-form";
import { OpenChatButton } from "@/components/customer/ask-buttons";

export default function NewTicketPage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-12">
      <h1 className="text-3xl font-semibold tracking-tight">Contact support</h1>
      <p className="mt-2 text-muted-foreground">
        Describe your issue and we’ll create a ticket. Many questions are answered faster in the chat.
      </p>
      <div className="mt-4">
        <OpenChatButton />
      </div>
      <div className="mt-8">
        <ContactForm />
      </div>
    </div>
  );
}
