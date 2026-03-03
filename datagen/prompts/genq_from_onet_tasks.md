## Task
Generate a *tool-use scenario with withheld information* grounded in workplace tasks performed by a given occupation.

## Objective

Brainstorm a workplace scenario in which {OCCUPATION} needs to perform *all of the following workplace tasks*, and analyze the provided MCP servers and their available tools to create
- a realistic user request that **deliberately omits 1–3 key parameters** an agent would need to ask about to complete the task
- a list of target tool calls that must be made to fulfill the request and their outputs
- the withheld parameters and the clarifying questions the agent should ask

## Workplace Tasks

**Occupation:** {OCCUPATION}
**Occupation Description:** {OCCUPATION_DESCRIPTION}
**Tasks:**
{TASKS}

Here are some search results related to the occupation and tasks. Use them to add constraints, context, and details to the request

{TASK_REFERENCES}

## MCP Servers

{SERVER_DESCRIPTIONS}

## Guidelines

### Scenario Brainstorming

- Think of realistic, specific scenarios where someone would need to use at least {NUM_TOOLS} target tools provided by the MCP servers to accomplish a meaningful task
- Consider diverse real-world contexts such as:
  - Content creators managing their online presence across different platforms
  - Researchers gathering and analyzing information from multiple sources
  - Developers building and deploying applications using different services
  - Business professionals managing projects and communications across platforms
  - Students working on complex assignments requiring multiple tools
  - Entrepreneurs launching new ventures using various services
- The scenario should be detailed and authentic, representing genuine use cases that span multiple services

### Request Realism

- Create requests that represent real-world scenarios where users would genuinely need the tools provided by the MCP servers
- The request should sound natural and authentic, as if asked by someone with a specific goal
- Include relevant context, constraints, and details that make the request engaging
- Consider workflows that require multiple complementary tools working together across different services
- Think about how different servers support each other in real-world use cases
- Use the search results to add constraints, context, and details to the request

### Information Withholding (Critical for this variant)

**Override the self-containment requirement**: Deliberately omit 1–3 key parameters from the request that the agent would naturally need to ask about before calling tools.

- Choose parameters that are **naturally unknown to the AI** — things a human user would know but might not think to specify upfront (e.g., account IDs, project names, date ranges, specific resource identifiers, usernames, file paths, API endpoints)
- Use **vague references** instead of specific values:
  - Instead of "account ID `acme-corp-2847`" → use "my account"
  - Instead of "January 2024" → use "last month" or "recently"
  - Instead of "project `frontend-redesign`" → use "the project I've been working on"
  - Instead of "repository `my-org/backend`" → use "my backend repo"
- The withheld parameters must be things the agent **cannot guess from context** — they require explicit confirmation from the user
- The request must still be **realistic**: a real user might genuinely forget to specify these values upfront
- The agent should be able to understand the intent but must ask before calling tools with the missing parameters
- Withhold **1 to 3** parameters — not so many that the request is incoherent

### Server and Target Tools Selection

- Select *at least {NUM_TOOLS} target tools* that work together
- The request should require a sequence or combination of tool calls to solve completely
- Choose target tools based on how they complement each other across different services/domains
- Consider each tool's description and purpose when crafting the cross-server workflow
- Ensure target tool calls create a logical, interconnected workflow

### Multi-Tool Integration

- Think about how different tools' capabilities can be combined
- Consider how data flows between tools and which **dependency patterns** connect them:
  - **Parameter dependency**: One tool's output provides input for the next (e.g., a lookup result feeds into a calculation)
  - **Conditional routing**: A tool's result determines which tool to call next (e.g., an inspection finding a violation triggers a reporting tool rather than a routine filing tool)
  - **Cross-validation**: Two tools verify or contradict each other's findings on the same request
  - **Aggregation**: Parallel tool calls whose results must be combined into a single response
- Create realistic scenarios where multiple tools need to work together
- Focus on complementary functionalities across different domains

### Request Complexity

- Create requests that are complex enough to warrant using at least {NUM_TOOLS} target tools across multiple servers
- The request should have multiple components or require several steps that span different services
- Include relevant context or constraints that make the multi-tool usage necessary
- Do not contain the exact target tool names or server names in the request
- Ensure the request cannot be reasonably fulfilled with tools from just a single server
- Create scenarios that naturally require different types of services working together

### Withheld Info and Follow-up Questions

- `withheld_info`: List the 1–3 parameters that were deliberately omitted from the request. For each, provide the parameter name, a brief description of why it's needed, and its actual value.
- `target_followup_questions`: Write the clarifying questions a well-behaved agent *should* ask the user to obtain the withheld parameters. Keep them natural and concise (one question can elicit multiple values if they're related).

### Output Format

Your response should include:
1. **Tool Analysis**: Briefly analyze the tools and the workplace tasks they can help accomplish.
2. **Cross-Tool Workflow**: Describe the workflow showing how tools will be used together, including the dependencies among tools and any decision points where intermediate results affect the next step.
3. **Withheld Info**: The parameters intentionally omitted from the request, with their actual values.
4. **Target Follow-up Questions**: The clarifying questions the agent should ask to obtain the withheld parameters.
5. **Target Tools**: The specific tools, their server names, their input arguments (using the actual withheld values), AND the output from executing the tools.
6. **Request**: A clear, realistic user request that uses vague references for the withheld parameters — the agent needs to ask for clarification before making tool calls.

## Output
Ensure your request is grounded in all {NUM_TASKS} workplace tasks and uses at least {NUM_TOOLS} tools to solve completely. Provide your response in the following JSON format:

Machine-readable schema source of truth: `prompts/genq_from_onet_tasks_withheld_output_schema.json`.

```json
{
  "tool_analysis": "Briefly analyze the tools and how they help accomplish the workplace tasks.",
  "cross_tool_workflow": "Describe the workflow: for each tool-to-tool link, state the dependency type (parameter dependency, conditional routing, cross-validation, or aggregation) and note any decision points where intermediate results change the next step.",
  "withheld_info": [
    {
      "parameter": "account_id",
      "description": "The user's account identifier needed to look up invoices and account status",
      "value": "acme-corp-2847"
    },
    {
      "parameter": "date_range",
      "description": "The time period for filtering records",
      "value": "January 2024"
    }
  ],
  "target_followup_questions": [
    "What is your account ID or company name?",
    "What date range are you interested in?"
  ],
  "target_tools": [
    {
      "server": "Server1",
      "tool": "get_account_status",
      "arguments": {"account_id": "acme-corp-2847"},
      "output": "Account status: Active, Plan: Professional, Last payment: 2024-01-15"
    },
    {
      "server": "Server2",
      "tool": "list_invoices",
      "arguments": {"account_id": "acme-corp-2847", "start_date": "2024-01-01", "end_date": "2024-01-31"},
      "output": "3 invoices found for January 2024: INV-001 ($1,200), INV-002 ($850), INV-003 ($2,100)"
    }
  ],
  "request": "Can you check my account status and pull up my invoices from last month? I need to reconcile some charges."
}
```

