"""LLM query rewriting for vocabulary mismatch.

Users describe problems in their own words ("the AI link so the chatbot stops"), while the docs
use product terms ("disable LLM integration, SearchUnifyGPT"). The LLM maps the question to doc
vocabulary with the help of a short product glossary. The retriever searches with the original
and the rewritten query and fuses the results, so a bad rewrite can't remove good matches.
"""

from functools import lru_cache

from supportpilot.llm.provider import get_llm

GLOSSARY = """SearchUnify product areas and terms:
- Content Sources: connectors that crawl and index data (Salesforce, Jira, Zendesk, Confluence, websites,
  files...), crawl frequency, re-connect, field mapping, Find and Replace rules, event subscription
- Search Clients: search UIs on Salesforce/Zendesk/websites, hosted search client, search client editor,
  facets, summary fields, session timeout, interface language, clone
- Search Tuning: keyword boosting, field-value boosting, auto boosting, auto tuning, intent tuning,
  NER, auto spell corrector, facet interpreter, rich snippets/cards
- NLP Manager: synonyms, stopwords, taxonomy, intents and utterances, Did You Mean
- Search Analytics: search report, searches with no result, click position, content gap analysis,
  case deflection, conversions, sessions/session journey, leadership dashboard, export to Tableau
- Alerts: search frequency alerts, analytics reports by email, indexing notifications
- Agentic Suite: AI agents (AI Support Agent, AI Knowledge Agent, AI Escalation Manager,
  AI Case Quality Auditor), MCPs, deploy on Salesforce
- Apps: Agent Helper (case summary, actions, Slack case swarming), Escalation Predictor, agent routing
- LLM Integrations: configure LLM, SearchUnifyGPT, LLM usage insights, error logs
- ML Workbench: content annotation, rich snippets
- End-user features: advanced search (wildcards), bookmarks, saved results, Chrome extension, password recovery
- Administration and users: add user, user role (admin/moderator), scope, SAML/SSO (Okta, Azure, OneLogin),
  admin logs, IP whitelisting, API client ID/secret, access/refresh tokens, SDKs (Python, Java)"""

PROMPT = """You turn a user's support question into ONE search query for the SearchUnify documentation.
Use the documentation's own terms from the glossary where they fit the user's meaning. Keep product
and connector names. 4-12 words, no explanation, no quotes.

{glossary}

User question: {question}
Search query:"""


@lru_cache(maxsize=2048)
def rewrite_query(question: str) -> str:
    llm = get_llm(reasoning_effort="low", max_tokens=600)
    out = llm.invoke(PROMPT.format(glossary=GLOSSARY, question=question)).content
    return str(out).strip().strip('"').split("\n")[0]
