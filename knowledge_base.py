"""
Knowledge base for the Fernhill Wealth support agent.

FERNHILL WEALTH IS FICTIONAL. It is modelled on how UK wealth-management firms
present their services, complaints procedures and client support, so that the
agent can be tested against realistic material without naming, quoting or
impersonating any real firm.

Everything below is invented and safe by construction:
  - The website lives on the `.example` domain, which is reserved by internet
    standard (RFC 2606) and can never belong to a real organisation.
  - Telephone numbers come from the ranges Ofcom sets aside for film and drama
    (020 7946 0xxx, 0117 496 0xxx, 0808 157 0xxx), so the agent can never hand
    a customer a real person's number.
  - The one real number quoted is the Financial Ombudsman Service, a public
    body whose contact details every UK regulated firm is required to give.

MAINTENANCE: each entry carries a `source` URL so an answer can be traced to
the page it came from. Keep `topic` strings stable; the retrieval tests rank
entries by them.
"""

KNOWLEDGE_BASE = [
    {
        "topic": "company overview who we are",
        "content": (
            "Fernhill Wealth is a UK wealth management firm offering financial planning "
            "and investment management to private clients, families and trustees. It was "
            "founded in Bristol in 1998 and now has six offices across England and "
            "Scotland. As at 31 December 2025 it managed £9.4 billion of client assets."
        ),
        "source": "https://www.fernhillwealth.example/about-us/",
    },
    {
        "topic": "main phone number general enquiries switchboard",
        "content": (
            "The main switchboard number is 0117 496 0100. Use this for general "
            "enquiries, for business enquiries, or if you are unsure who to speak to. "
            "Lines are open 08:30 to 18:00, Monday to Friday. Calls may be recorded for "
            "regulatory purposes."
        ),
        "source": "https://www.fernhillwealth.example/contact-us/",
    },
    {
        "topic": "booking an appointment becoming a new client",
        "content": (
            "People considering becoming a client can book a free introductory "
            "appointment through the Book an Appointment page, or request a call back at a "
            "convenient time. The new client enquiry line is 0808 157 0450, which is free "
            "to call from UK landlines and mobiles."
        ),
        "source": "https://www.fernhillwealth.example/book-an-appointment/",
    },
    {
        "topic": "callback times when you can be called",
        "content": (
            "When requesting a call back, the times offered on the booking form run from "
            "09:00 to 17:00 on the hour, with a final 17:45 slot. The form also asks for "
            "your preferred date. Note these are callback slots, not office opening hours."
        ),
        "source": "https://www.fernhillwealth.example/contact-us/",
    },
    {
        "topic": "client portal login and access problems",
        "content": (
            "Existing clients log in through the client portal link on the website. If you "
            "are having trouble getting into the portal, there is a dedicated portal access "
            "helpline on 0117 496 0233, open 09:00 to 17:30 on working days."
        ),
        "source": "https://www.fernhillwealth.example/client-support/portal/",
    },
    {
        "topic": "updating personal details account queries statements valuations",
        "content": (
            "For account matters such as updating your personal or financial details, "
            "requesting statements or valuations, checking your portfolio balance or recent "
            "transactions, or asking about your fee structure, the recommended route is to "
            "speak to your own adviser directly. If you do not know who your adviser is, "
            "call 0117 496 0100 and the team will connect you."
        ),
        "source": "https://www.fernhillwealth.example/client-support/",
    },
    {
        "topic": "how to make a complaint",
        "content": (
            "Complaints can be made by letter, telephone, email or in person, and may be "
            "raised with anyone in the firm. The Client Resolution team can be reached by "
            "email at complaints@fernhillwealth.example, by phone on 0117 496 0355, or by "
            "post to Client Resolution, Fernhill Wealth, Fernhill House, 22 Harbourside "
            "Walk, Bristol, BS1 5XY. If someone is complaining on your behalf, written "
            "permission is required. There is no charge for making a complaint."
        ),
        "source": "https://www.fernhillwealth.example/complaints/",
    },
    {
        "topic": "complaint timescales what happens next ombudsman",
        "content": (
            "An acknowledgement letter is issued within five working days. After "
            "investigation, a Final Response Letter setting out the findings is sent. If the "
            "investigation cannot be completed within eight weeks, the firm writes to "
            "explain the delay and outline your options. If you remain dissatisfied, or if "
            "eight weeks pass without resolution, you can refer the matter free of charge to "
            "the Financial Ombudsman Service on 0800 023 4567."
        ),
        "source": "https://www.fernhillwealth.example/complaints/",
    },
    {
        "topic": "financial planning service what it covers",
        "content": (
            "Financial planning covers either a comprehensive plan across your whole "
            "financial position, or one-off advice on a single topic. Areas include pensions "
            "and retirement planning, tax planning, financial protection, inheritance tax "
            "and estate planning, cashflow modelling, investing for children, and support "
            "through life-changing events such as divorce or long-term care. All financial "
            "planners either hold or are working towards Chartered status."
        ),
        "source": "https://www.fernhillwealth.example/services/financial-planning/",
    },
    {
        "topic": "office locations bristol local offices",
        "content": (
            "There are six offices: Bristol, London, Manchester, Edinburgh, Leeds and "
            "Birmingham, listed on the Offices page. The Bristol head office is at Fernhill "
            "House, 22 Harbourside Walk, Bristol, BS1 5XY, a short walk from Temple Meads "
            "station. Advisers are also available for telephone and video meetings."
        ),
        "source": "https://www.fernhillwealth.example/offices/bristol/",
    },
    {
        "topic": "closing or transferring an account",
        "content": (
            "If you are thinking about closing your account or transferring investments "
            "elsewhere, the firm asks that you speak to your adviser first. They can review "
            "your fee structure, discuss investment performance, compare offers, or "
            "reconsider your financial plan before you decide."
        ),
        "source": "https://www.fernhillwealth.example/client-support/",
    },
    {
        "topic": "bereavement divorce power of attorney life events support",
        "content": (
            "There is a dedicated help and support route for clients dealing with "
            "bereavement, divorce, serious illness, or arranging power of attorney. These "
            "situations are handled by a specialist team rather than through the standard "
            "enquiry channels."
        ),
        "source": "https://www.fernhillwealth.example/client-support/life-events/",
    },
]
