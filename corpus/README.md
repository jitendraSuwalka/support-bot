# Fernhill Wealth corpus

The documents in this folder are the knowledge base the agent retrieves from.
**Fernhill Wealth is fictional.** Every policy, fee, address, person and phone
number here is invented; the firm's website lives on the reserved `.example`
domain and its phone numbers come from the ranges Ofcom sets aside for drama.
The Financial Services Register number and company number are deliberately
omitted rather than invented, so they cannot collide with a real firm's.

The content is modelled on how UK wealth-management firms present their
services, so that retrieval and evaluation are tested against realistic
material. The first draft was reviewed against primary sources by a separate
AI reviewer briefed as a compliance officer; that review found factual, terminology and
consistency faults which were corrected on 20 October 2026, and three documents
were added for topics clients commonly ask about (tax outside wrappers,
becoming a client, cash and dealing).

Where a document states a **public regulatory fact**, the figure is real and
was checked against a primary source on 20 October 2026:

| Fact | Figure | Source |
|---|---|---|
| ISA allowance, 2026/27 | £20,000 | gov.uk, *Individual Savings Accounts* |
| Cash ISA cap, investors under 65 | £12,000 from 6 April 2027; no S&S→cash transfers; charge on cash interest in S&S ISAs | gov.uk, *Tax-free savings newsletter 19* |
| Lifetime ISA | £4,000 a year, 25% bonus, ages 18–39; first-time-buyer ISA under consultation | gov.uk |
| ISA transfer timescales | 15 working days (cash to cash); 30 calendar days (other) | gov.uk, *Transferring your ISA* |
| Pension annual allowance | £60,000 (taper floor £10,000) | gov.uk, *Pension schemes rates* |
| Money purchase annual allowance | £10,000 | gov.uk, *Pension schemes rates* |
| Lump sum allowance | £268,275 | gov.uk, *Pension schemes rates* |
| Lump sum and death benefit allowance | £1,073,100 | gov.uk, *Pension schemes rates* |
| Pensions and inheritance tax | unused funds and death benefits in the estate for deaths from 6 April 2027 | gov.uk, *Inheritance tax: unused pension funds and death benefits* |
| Normal minimum pension age | 55, rising to 57 on 6 April 2028 | gov.uk |
| Defined benefit transfer advice | required above £30,000 | Pension Schemes Act 2015, s.48 |
| Capital gains tax | annual exempt amount £3,000; rates 18% / 24% | gov.uk, *Capital Gains Tax rates and allowances* |
| Dividends | allowance £500; rates 10.75% / 35.75% / 39.35% | gov.uk, *Tax on dividends* |
| FSCS: investments | £85,000 per person, per firm | fscs.org.uk |
| FSCS: deposits | £120,000 per person, per banking licence (from 1 Dec 2025) | fscs.org.uk |
| Complaints | final response within 8 weeks; referral within 6 months; 6-year / 3-year limits | financial-ombudsman.org.uk |
| Financial Ombudsman Service | 0800 023 4567 / 0300 123 9123, Exchange Tower, London, E14 9SR | financial-ombudsman.org.uk |
| Fraud reporting | Report Fraud, 0300 123 2040; Scotland via 101 | reportfraud.police.uk |
| Call-recording retention | at least five years | FCA Handbook, SYSC 10A |

Regulatory figures change; the `last_reviewed` date in each file's front matter
says when its content was last checked.

## Document format

Each file is Markdown with a YAML front-matter block:

```yaml
---
title: Fees and charges
url: https://www.fernhillwealth.example/fees-and-charges/
version: 2026-10
last_reviewed: 2026-10-20
owner: Compliance
---
```

`rag/corpus.py` loads the folder, validates the front matter, and splits each
document into retrieval chunks along its `##` headings. The `title`, `url` and
heading travel with every chunk, so an answer can always be traced to the page
and section it came from.
