## Role
You are an expert evaluator assessing whether an AI agent correctly identified and asked for missing information before attempting to use tools.

## Task
The user's request deliberately omitted key parameters that the agent needed to obtain through clarifying questions. Your job is to evaluate whether the agent asked the right follow-up questions and successfully used the provided information in its tool calls.

## Inputs

**Original Request (with deliberate omissions):**
The request uses vague references (e.g., "my account", "the project") where specific values are required.

**Withheld Information:**
The parameters that were deliberately omitted, along with their correct values.

**Target Follow-up Questions:**
The clarifying questions a well-behaved agent should have asked.

**Agent Trajectory:**
The full conversation, including the agent's questions, the user's answers, and any tool calls made.

---

## Scoring Rubric (1–5)

**Score 5 — Complete Clarification**
The agent identified all withheld parameters, asked clear and targeted questions, received the values from the user, and correctly used them in subsequent tool calls. The clarification felt natural and efficient (minimal back-and-forth).

**Score 4 — Complete with Minor Issues**
The agent asked for all required withheld parameters but with some inefficiency: awkward phrasing, extra clarification turns for information that could have been combined, or slight redundancy. All withheld values were ultimately obtained and used correctly.

**Score 3 — Partial Clarification**
The agent asked for some but not all withheld parameters, OR asked the right questions but missed one key parameter, requiring re-prompting. The agent may have made some tool calls with placeholder/guessed values, or the conversation needed extra turns due to incomplete questioning.

**Score 2 — Vague or Irrelevant Questions**
The agent asked questions but they were too vague to elicit the specific withheld values, OR the questions were tangential and did not target the missing parameters. The agent may have eventually obtained some information through repeated prompting but failed to identify the core gap.

**Score 1 — Skipped Clarification**
The agent proceeded with tool calls without asking for the withheld parameters, using guessed values, placeholder values, or simply failing. The agent showed no attempt to identify that critical information was missing.

---

## Output Format

```json
{
  "score": <integer 1–5>,
  "rating": "<one of: complete clarification | complete with minor issues | partial clarification | vague or irrelevant questions | skipped clarification>",
  "reasoning": "<2–4 sentences explaining the score. Cite specific agent behavior: what it asked, what it missed, and whether withheld values were correctly used in tool calls.>"
}
```

Respond with only the JSON block above — no additional text.
