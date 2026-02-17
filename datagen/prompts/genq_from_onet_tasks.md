## Task
Generate a **Tool Use Question** grounded in specific workplace tasks performed by a given occupation.

## Objective
You are given a set of workplace tasks that a **{OCCUPATION}** performs, along with tools from MCP servers that can help accomplish those tasks. Your job is to craft a realistic user question that naturally requires using **all {NUM_TASKS} tasks** as the basis for a multi-step workflow, leveraging the provided tools as the means to accomplish them.

## Occupation Context
**{OCCUPATION}**: {OCCUPATION_DESCRIPTION}

### Knowledge Domains
{KNOWLEDGE}

### Skills
{SKILLS}

## Workplace Tasks
The following O*NET workplace tasks define the scenario. Your question must be grounded in these tasks:
{TASKS}

## Available Tools
The following tools are available from MCP servers matched to the tasks above. Use them as the means to accomplish the workplace tasks:

{TOOL_DESCRIPTIONS}

## Guidelines

### Scenario Grounding
- The scenario **must be grounded in the listed workplace tasks** — they define the work being done
- The tools are the **means** to accomplish the tasks, not the starting point
- Think about how a {OCCUPATION} would naturally encounter a situation requiring these tasks together
- Use the occupation's knowledge domains and skills to add realistic detail and context

### Question Realism
- Create a question that represents a real-world scenario where a {OCCUPATION} genuinely needs to perform these tasks
- The question should sound natural and authentic, as if asked by someone with a specific goal
- Include relevant context, constraints, and details that make the question engaging
- Draw on the occupation's knowledge domains and skills to make the scenario authentic

### Tool Integration
- Each tool should serve as a means to accomplish one or more of the workplace tasks
- Consider how data flows between tools (e.g., output from one tool feeds into another)
- The tools should create a logical, interconnected workflow that addresses the tasks
- Consider each tool and its parent MCP server's descriptions when crafting the workflow
- Do not include exact tool names or server names in the question

### Question Complexity
- The question should have multiple components corresponding to the workplace tasks
- Include relevant context or constraints that make multi-tool usage necessary
- Create scenarios that consist of several complementary tasks to accomplish an overall goal

### Output Format
Your response should include:
1. **Tool Analysis**: Briefly analyze the tools and the workplace tasks they can help accomplish.
2. **Cross-Tool Workflow**: Describe the workflow showing how tools will be used together to accomplish the workplace tasks.
3. **Target Tasks**: The specific workplace tasks and their task IDs that the question addresses.
4. **Target Tools**: The specific tools, their server names, and their input arguments that must be used, in the order they would likely be called. The input arguments MUST follow the tool's Input Schema exactly — do not invent or rename parameters.
5. **Question**: A clear, realistic user question that requires tool usage to accomplish the workplace tasks.

## Output
Ensure your question is grounded in all {NUM_TASKS} workplace tasks and uses the available tools to accomplish them. Provide your response in the following JSON format:

```json
{
  "tool_analysis": "Briefly analyze the tools and how they help accomplish the workplace tasks. If more than one ",
  "cross_tool_workflow": "Describe the workflow showing how tools will be used together to accomplish the workplace tasks.",
  "target_tasks": [
    {"id": "8823", "description": "Direct or coordinate an organization's financial or budget activities to fund operations, maximize investments, or increase efficiency."},
    {"id": "8824", "description": "Confer with board members, organization officials, or staff members to discuss issues, coordinate activities, or resolve problems."}
  ],
  "target_tools": [
    {"server": "Server1", "tool": "get_weather", "arguments": {"location": "Paris, France"}},
    {"server": "Server2", "tool": "send_email", "arguments": {"to": "bob@email.com", "body": "Hi bob"}}
  ],
  "question": "A clear, realistic user question grounded in the workplace tasks that requires multi-tool usage."
}
```
