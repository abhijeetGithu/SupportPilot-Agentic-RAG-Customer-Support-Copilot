// Mirrors the FastAPI schemas in src/supportpilot/api/schemas.py and tickets/schemas.py.

export type Priority = "low" | "medium" | "high" | "urgent";
export type TicketStatus = "open" | "in_progress" | "escalated" | "resolved" | "closed";
export type TicketSource = "chat" | "form" | "mcp";
export type AuthorType = "customer" | "ai" | "staff";
export type SessionMode = "ai" | "human";

export interface Citation {
  n: number; // the [n] used in the answer text
  title: string;
  url: string;
  heading: string | null;
}

export interface Ticket {
  id: number;
  number: string;
  subject: string;
  description: string;
  customer_name: string;
  customer_email: string;
  category: string | null;
  priority: Priority;
  confidence: number | null;
  priority_confidence: number | null;
  classifier_model: string | null;
  needs_review: boolean;
  status: TicketStatus;
  source: TicketSource;
  assignee: string | null;
  created_at: string;
  updated_at: string;
}

export interface TicketMessage {
  id: number;
  author_type: AuthorType;
  body: string;
  citations: Citation[] | null;
  created_at: string;
}

export interface TicketEvent {
  id: number;
  type: string;
  actor: string;
  payload: Record<string, unknown> | null;
  created_at: string;
}

export interface TicketDetail {
  ticket: Ticket;
  messages: TicketMessage[];
  events: TicketEvent[];
}

export interface TicketPage {
  items: Ticket[];
  total: number;
  page: number;
  page_size: number;
}

export interface CustomerTicketView {
  number: string;
  subject: string;
  status: TicketStatus;
  created_at: string;
  updated_at: string;
  messages: TicketMessage[];
}

export interface TicketDraft {
  subject: string;
  description: string;
}

export interface ChatMessage {
  role: "customer" | "ai" | "staff";
  content: string;
  id: number | null; // ticket message id for staff replies (used for polling)
}

export interface ChatHistory {
  session_id: string;
  mode: SessionMode;
  ticket_number: string | null;
  messages: ChatMessage[];
  pending_draft: TicketDraft | null;
}

export interface StaffMessages {
  mode: SessionMode;
  ticket_number: string | null;
  messages: ChatMessage[];
}

export interface SuggestedReply {
  body: string;
  citations: Citation[];
}

export interface Metrics {
  tickets_by_status: Record<string, number>;
  tickets_by_category: Record<string, number>;
  tickets_by_priority: Record<string, number>;
  needs_review: number;
  chat_sessions: number;
  ai_resolution_rate: number;
}

// Server-sent events from POST /api/chat/{sid}/stream and /resume (src/supportpilot/api/streaming.py)
export type ChatEvent =
  | { type: "token"; text: string }
  | { type: "tool_start"; tool: string; query?: string }
  | { type: "tool_end"; tool: string }
  | { type: "citations"; citations: Citation[] }
  | { type: "ticket_draft"; draft: TicketDraft } // followed by "done": wait for approve / edit / cancel
  | { type: "ticket_created"; number: string; category: string | null; priority: Priority; duplicate: boolean }
  | { type: "handoff"; number: string }
  | { type: "stored"; message_id: number } // human mode: message saved for the support team
  | { type: "done"; ticket_number: string | null }
  | { type: "error"; message: string };
