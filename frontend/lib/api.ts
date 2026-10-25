// Typed client for the FastAPI backend. All calls go to /api/*, which next.config.ts proxies to FastAPI.

import type {
  Attachment,
  ChatEvent,
  ChatHistory,
  CustomerTicketView,
  Metrics,
  StaffMessages,
  SuggestedReply,
  Ticket,
  TicketDetail,
  TicketPage,
} from "@/lib/types";

const TOKEN_KEY = "supportpilot.staffToken";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export const staffToken = {
  get: () => (typeof window === "undefined" ? null : localStorage.getItem(TOKEN_KEY)),
  set: (t: string) => localStorage.setItem(TOKEN_KEY, t),
  clear: () => localStorage.removeItem(TOKEN_KEY),
};

async function request<T>(path: string, init: RequestInit & { staff?: boolean } = {}): Promise<T> {
  const headers = new Headers(init.headers);
  // JSON bodies only: for FormData (uploads) the browser sets multipart/form-data with its boundary
  if (typeof init.body === "string" && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  if (init.staff) {
    const token = staffToken.get();
    if (token) headers.set("Authorization", `Bearer ${token}`);
  }
  const res = await fetch(`/api${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {}
    if (res.status === 401 && init.staff) staffToken.clear();
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown) => JSON.stringify(body);

// ---------- streaming (SSE over POST) ----------

/** POSTs to an SSE endpoint and yields parsed ChatEvents as they arrive. */
export async function* streamEvents(path: string, body: unknown, signal?: AbortSignal): AsyncGenerator<ChatEvent> {
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: json(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {}
    throw new ApiError(res.status, detail);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const data = block
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trimStart())
        .join("\n");
      if (data) yield JSON.parse(data) as ChatEvent;
    }
  }
}

// ---------- customer ----------

export const api = {
  createSession: (name: string, email: string) =>
    request<{ session_id: string; mode: "ai" | "human" }>("/sessions", { method: "POST", body: json({ name, email }) }),
  history: (sid: string) => request<ChatHistory>(`/chat/${sid}/history`),
  staffMessages: (sid: string, after: number) => request<StaffMessages>(`/chat/${sid}/messages?after=${after}`),
  chat: (sid: string, message: string, attachmentIds: string[] = [], signal?: AbortSignal) =>
    streamEvents(`/chat/${sid}/stream`, { message, attachment_ids: attachmentIds }, signal),
  /** One file per request; chat uploads pass the session id, contact-form uploads don't. */
  upload: (file: File, sessionId?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (sessionId) form.append("session_id", sessionId);
    return request<Attachment>("/uploads", { method: "POST", body: form });
  },
  resume: (
    sid: string,
    decision: { action: "approve" | "edit" | "cancel"; subject?: string; description?: string },
    signal?: AbortSignal,
  ) => streamEvents(`/chat/${sid}/resume`, decision, signal),

  createTicket: (body: {
    customer_name: string;
    customer_email: string;
    subject: string;
    description: string;
    attachment_ids?: string[];
  }) =>
    request<{ ticket: Ticket; duplicate_of_existing: boolean }>("/tickets", { method: "POST", body: json(body) }),
  lookupTicket: (number: string, email: string) =>
    request<CustomerTicketView>("/tickets/lookup", { method: "POST", body: json({ number, email }) }),
  customerReply: (number: string, email: string, body: string) =>
    request<CustomerTicketView>(`/tickets/${encodeURIComponent(number)}/replies`, {
      method: "POST",
      body: json({ email, body }),
    }),

  // ---------- staff ----------
  login: (username: string, password: string) =>
    request<{ access_token: string }>("/auth/login", { method: "POST", body: json({ username, password }) }),
  tickets: (params: Record<string, string | number | boolean | undefined>) => {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== "") qs.set(k, String(v));
    return request<TicketPage>(`/admin/tickets?${qs}`, { staff: true });
  },
  ticket: (id: number) => request<TicketDetail>(`/admin/tickets/${id}`, { staff: true }),
  updateTicket: (id: number, changes: Partial<Pick<Ticket, "status" | "priority" | "category" | "assignee">>) =>
    request<TicketDetail>(`/admin/tickets/${id}`, { method: "PATCH", body: json(changes), staff: true }),
  staffReply: (id: number, body: string) =>
    request<TicketDetail>(`/admin/tickets/${id}/replies`, { method: "POST", body: json({ body }), staff: true }),
  suggestReply: (id: number) =>
    request<SuggestedReply>(`/admin/tickets/${id}/suggest-reply`, { method: "POST", staff: true }),
  escalations: () => request<TicketPage>("/admin/escalations", { staff: true }),
  takeEscalation: (id: number) =>
    request<TicketDetail>(`/admin/escalations/${id}/take`, { method: "POST", staff: true }),
  metrics: () => request<Metrics>("/admin/metrics", { staff: true }),
  /** Staff-only download (needs the bearer token, so fetched as a blob rather than a plain link). */
  attachmentBlob: async (id: string) => {
    const res = await fetch(`/api/admin/attachments/${id}`, {
      headers: { Authorization: `Bearer ${staffToken.get() ?? ""}` },
    });
    if (!res.ok) throw new ApiError(res.status, res.statusText);
    return res.blob();
  },
};
