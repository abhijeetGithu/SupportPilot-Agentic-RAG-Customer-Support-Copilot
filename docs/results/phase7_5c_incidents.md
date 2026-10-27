# Phase 7.5c: Incident detection (spotting outages from ticket bursts)

When a platform problem hits, many customers report it within minutes, each in their own words, and
each ticket looks routine on its own. SupportPilot now notices the pattern, groups the reports into one
**incident**, lets staff answer everyone at once, and tells the next affected customer straight away.

## How it works
Runs once per new ticket (chat, contact form or MCP), inside `TicketService.create`; a failure never
blocks ticket creation.

```
new ticket ─► bge embedding (stored on the ticket)
           ─► is it a problem report? (failure language: error, 401, slow, not running, no data...)
                 no ─► done (how-to questions never open or join incidents)
           ─► similar to an active incident's centroid AND to ≥2 of its tickets? ─► join (+ auto-reply
                 with the staff notice, if one is set)
           ─► else: similar problem reports in the last 2 h from ≥3 different customers, and a burst?
                 Poisson test: λ = similar tickets in the previous 7 days scaled to 2 h (floor 0.05)
                 p = P(X ≥ customers | λ) ≤ 0.01 ─► open an incident
```

Why each rule exists (all found by measuring, below):
| Rule | Without it |
|---|---|
| Similarity 0.76 (bge cosine) | lower values merged neighbouring outages; pairwise calibration below |
| Distinct customers | one customer repeating themselves looked like a burst |
| **Problem-report gate** | every false incident came from bursts of similar **how-to** questions; similarity alone can't tell "how do I add Jira?" asked six times from an outage (median pair similarity 0.78 vs 0.80) |
| Poisson burst test vs. 7-day baseline | a team that always gets Salesforce questions would get an "incident" every day |
| Join needs centroid + 2 member matches | the centroid drifted toward "search problems in general" and swallowed a neighbouring outage |

## Staff and customers
- **Incidents page** (sidebar badge with the active count): AI title and summary for staff (generated on
  first open, never shown to customers), a plain-language reason ("3 customers in 120 minutes; usually
  0.05; chance by coincidence under 0.01%"), the linked tickets with a remove button for wrong links.
- **Customer notice:** staff text that the assistant shows customers who describe the problem, and that
  new matching tickets receive as an instant reply.
- **Update everyone:** one message posted to every open ticket in the incident.
- **Resolve:** optional final message, and the open tickets are resolved together.
- **Ticket page:** a banner "Part of active incident INC-00001" linking to it.
- **Chat:** the staff notice is streamed as its own event and shown **verbatim** in a "Known issue" card;
  the assistant only acknowledges it. First version let the LLM restate the notice, and it turned "next
  update by 3 PM" into "should be live by 3 PM" and attached a doc citation to it. Showing the staff text
  directly removes that risk.

## Calibration and evaluation (`python evals/run_incident_eval.py [--sweep] [--data ...]`)
**Pairwise similarity** (bge, synthetic reports): same outage, different customers: median 0.80;
same product area but different issue: 0.56; background questions: 0.57.

**Traffic replay.** 3 days per seed × 5 seeds: 150 real SearchUnify questions as background, a popular
how-to question 6 times across the days, near-miss groups (3 same-area questions within 45 min), and 4
outages × 7 reports with exponential arrivals (mean 8 min). No LLM calls.

| Version (tuning set, 20 outages) | Detected | False incidents | Member precision | Member recall |
|---|---|---|---|---|
| Similarity only, 0.74 | 8/20 | 5 | 0.68 | 0.80 |
| + problem-report gate | 12/20 | 0 | 0.70 | 0.88 |
| + join needs member matches, 0.74 | 18/20 | 0 | 0.89 | 0.79 |
| **+ similarity 0.76 (chosen)** | **19/20** | **0** | **0.95** | **0.79** |

**Held-out set** (3 new outages, 2 new near-miss groups, a new popular question; written after all
choices above, run once): **10/15 detected, 0 false incidents**, member precision 1.0. The miss was one
outage in all 5 seeds ("all crawls stuck in Queued"): its reports used phrasings the gate didn't cover
("did not run", "never runs", "not starting") and varied more (median pair similarity 0.73). Adding
generic negation phrases to the gate gave **15/15, 0 false, precision 1.0, recall 0.64, median 35 min
to detection**; that number is no longer held-out. The gate still passes only 10 of 232 how-to golden
questions and 0 of 27 near-miss/how-to questions.

Typical detection: at the **3rd customer's report**, median **21 minutes** after the first (tuning set).

Limits: synthetic reports written by one person; outages described in very different words are the weak
spot (lower recall); the problem gate is a rule list (a small classifier would generalise better); the
window scan is linear (fine for thousands of tickets a week, a vector index beyond that).

## End to end (real embeddings and LLM, isolated server on a copy of the database)
- 3 customers filed Salesforce 401 reports through the contact form; the 3rd opened **INC-00001** and
  linked all three. AI title: "Salesforce crawls failing with HTTP 401".
- Staff saved a notice and sent an update to the 3 customers.
- A 4th customer asked the chat; the "Known issue" card showed the notice verbatim and the assistant
  said the next update was expected by 3 PM, without inventing workarounds.
- The 4th customer's form ticket joined the incident and immediately got the notice as a reply; an
  unrelated "analytics shows no data" ticket filed at the same time did not join.
- Resolving with a final message resolved all 4 tickets; the unrelated ticket stayed open.

| Staff incident page | Customer chat: known issue |
|---|---|
| ![Incident](../images/incident.png) | ![Known issue in chat](../images/chat-known-issue.png) |

## Tests
12 new tests (`tests/test_incidents.py`, topic-vector embeddings): problem gate; burst from 3 customers
opens an incident; one customer repeating doesn't; how-to questions don't; a usual rate of similar
reports isn't a burst; later reports join and get the notice; detection failures never block tickets;
console (list, notice, update to all, customer sees it), remove a wrong ticket and resolve; AI summary;
the assistant gets only staff text (generic line before a notice exists); the notice is streamed as its
own chat event. Full suite: **80 passed**.
