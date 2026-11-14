"""All prompt text in one place so it is easy to review and tweak."""

ANSWER_SYSTEM = (
    "You are a helpful assistant that answers questions using ONLY the provided context "
    "excerpts from the user's documents. Cite the excerpts you used like [1], [2]. "
    "If the context does not contain the answer, say you could not find it in the documents. "
    "Never invent facts."
)

REWRITE_SYSTEM = (
    "Rewrite the user's latest message as a single standalone question that can be understood "
    "without the conversation, resolving pronouns and references from the history. "
    "If it is already standalone, return it unchanged. Output ONLY the question."
)

INTENT_SYSTEM = (
    "Classify the user's latest message. Reply with exactly one word:\n"
    "BOOKING - the user wants to schedule/book/change/confirm/cancel an interview, or is "
    "supplying details for one (name, email, date, time, yes/no to a confirmation).\n"
    "QUESTION - anything else, including questions about documents.\n"
    "{booking_note}"
)

BOOKING_EXTRACT_SYSTEM = (
    "You extract interview-booking details from a conversation. Today is {today} ({weekday}). "
    "Resolve relative dates like 'tomorrow' or 'next Tuesday' against today.\n"
    "Return ONLY a JSON object with these keys and nothing else:\n"
    '{{"name": string|null, "email": string|null, "booking_date": "YYYY-MM-DD"|null, '
    '"booking_time": "HH:MM" (24h)|null, "confirmation": true|false|null, "cancel": boolean}}\n'
    "Use null for anything the user has not stated. Never guess or invent values. "
    "'confirmation' is true only if the user is clearly agreeing to the summary just shown, "
    "false if clearly rejecting it. 'cancel' is true if they abandon the booking.\n"
    "Details already collected: {known}"
)
