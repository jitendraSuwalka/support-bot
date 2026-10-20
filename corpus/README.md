# Fernhill Wealth corpus

The documents in this folder are the knowledge base the agent retrieves from.
**Fernhill Wealth is fictional.** Every policy, fee, address, person and phone
number here is invented; the firm's website lives on the reserved `.example`
domain and its phone numbers come from the ranges Ofcom sets aside for drama.

The content is modelled on how UK wealth-management firms actually present
their services, so that retrieval and evaluation are tested against realistic
material. Where a document states a **public regulatory fact**, the figure is
real and was checked against a primary source on 20 October 2026:

| Fact | Figure | Source |
|---|---|---|
| ISA allowance, 2026/27 | £20,000 | HMRC / provider guidance |
| Cash ISA cap for under-65s | £12,000 from 6 April 2027 | HM Treasury announcement |
| Lifetime ISA | £4,000 a year, 25% bonus, ages 18–39 | gov.uk |
| Pension annual allowance | £60,000 | gov.uk, *Pension schemes rates* |
| Money purchase annual allowance | £10,000 | gov.uk, *Pension schemes rates* |
| Lump sum allowance | £268,275 | gov.uk, *Pension schemes rates* |
| FSCS: investments | £85,000 per person, per firm | fscs.org.uk |
| FSCS: deposits | £120,000 per person, per bank (from 1 Dec 2025) | fscs.org.uk |
| Complaint final response | 8 weeks | financial-ombudsman.org.uk |
| Referral to the Ombudsman | within 6 months of the final response | financial-ombudsman.org.uk |
| Financial Ombudsman Service | 0800 023 4567, Exchange Tower, London, E14 9SR | financial-ombudsman.org.uk |

Regulatory figures change; the `last_reviewed` date in each file's front matter
says when its content was last checked.

## Document format

Each file is Markdown with a YAML front-matter block:

```yaml
---
title: Fees and charges
url: https://www.fernhillwealth.example/fees-and-charges/
version: 2026-10
last_reviewed: 2026-10-01
owner: Compliance
---
```

`rag/corpus.py` loads the folder, validates the front matter, and splits each
document into retrieval chunks along its `##` headings. The `title`, `url` and
heading travel with every chunk, so an answer can always be traced to the page
and section it came from.
