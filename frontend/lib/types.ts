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
  incident_id: number | null;
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

export interface Attachment {
  id: string;
  filename: string;
  content_type: string;
  kind: "image" | "pdf" | "text";
  size_bytes: number;
  uploaded_by: string;
  analysis: FileAnalysis | null;
  created_at: string;
}

/** What the vision model / PDF reader found in a screenshot or PDF. */
export interface FileAnalysis {
  summary: string;
  visible_text: string;
  error_messages: string[];
  product_area: string;
  contains_instructions: boolean;
  source: "vision" | "pdf_text" | "pdf_vision";
  model: string | null;
}

export interface AttachmentMeta {
  id: string;
  filename: string;
  kind: string;
  size_bytes: number;
}

export interface TicketDetail {
  ticket: Ticket;
  messages: TicketMessage[];
  events: TicketEvent[];
  attachments: Attachment[];
  incident: { id: number; number: string; title: string; status: IncidentStatus } | null;
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
  attachments: Attachment[];
}

export interface TicketDraft {
  subject: string;
  description: string;
}

export interface ChatMessage {
  role: "customer" | "ai" | "staff";
  content: string;
  id: number | null; // ticket message id for staff replies (used for polling)
  attachments?: AttachmentMeta[];
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
  | { type: "known_issue"; number: string; notice: string }
  | {
      type: "grounding";
      verdict: "grounded" | "revised" | "no_claims" | "unchecked";
      claims?: number;
      supported?: number;
      unsupported?: string[];
      revised_text?: string;
    }
  | { type: "output_guard"; reasons: string[]; text: string }
  | { type: "stored"; message_id: number } // human mode: message saved for the support team
  | { type: "done"; ticket_number: string | null }
  | { type: "error"; message: string };

// ---------- knowledge loop ----------
export type KbStatus = "draft" | "published" | "rejected";

export interface KbArticle {
  id: number;
  number: string;
  source_ticket_id: number | null;
  source_ticket_number: string | null;
  title: string;
  body: string;
  product_area: string;
  keywords: string[];
  status: KbStatus;
  doc_coverage: { title: string; url: string; score: number } | null;
  generalizable: boolean;
  draft_note: string;
  model: string | null;
  reviewed_by: string | null;
  url: string;
  linked_in_replies: number;
  created_at: string;
  updated_at: string;
  published_at: string | null;
}

/** A published article an agent can link in a reply; score = similarity to the ticket or search text. */
export interface KbSuggestion {
  id: number;
  number: string;
  title: string;
  product_area: string;
  url: string;
  score: number;
}

export interface KbPublicArticle {
  number: string;
  title: string;
  body: string;
  product_area: string;
  published_at: string | null;
}

export interface GapTopic {
  label: string;
  count: number;
  sessions: number;
  outcomes: Partial<Record<"answered" | "ticket" | "handoff" | "error", number>>;
  avg_coverage: number | null;
  closest_doc: { title: string; url: string } | null;
  last_seen: string;
  examples: string[];
}

export interface GapReport {
  days: number;
  threshold: number;
  questions: number;
  gap_questions: number;
  ended_in_ticket_or_handoff: number;
  topics: GapTopic[];
}

// ---------- incidents ----------
export type IncidentStatus = "active" | "resolved";

export interface Incident {
  id: number;
  number: string;
  title: string;
  summary: string;
  status: IncidentStatus;
  public_note: string;
  tickets: number;
  customers: number;
  open_tickets: number;
  detection: {
    customers?: number;
    tickets?: number;
    baseline_similar?: number;
    expected_per_window?: number;
    p_value?: number;
    window_minutes?: number;
    opened_at?: string;
  };
  first_seen: string;
  last_seen: string;
  created_at: string;
  resolved_at: string | null;
  resolved_by: string | null;
}

export interface IncidentDetail {
  incident: Incident;
  tickets: Ticket[];
}
