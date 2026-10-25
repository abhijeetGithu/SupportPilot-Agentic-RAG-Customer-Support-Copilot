"""Phase 7: MCP server exposing SupportPilot tools to Claude Desktop / Cursor.

Run:  python -m supportpilot.mcp_server.server

Tools (all ticket changes go through TicketService with source=mcp):
- search_docs(query)
- create_ticket(customer_name, customer_email, subject, description)
- get_ticket_status(ticket_number)
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("supportpilot")

if __name__ == "__main__":
    mcp.run()
