## Task
Generate a **Tool Use Question** grounded in specific workplace tasks performed by a given occupation.

## Objective
You are given a set of workplace tasks that a **{OCCUPATION}** performs, along with MCP servers providing tools that can help accomplish those tasks. Your job is to craft a realistic user question that naturally requires using **{NUM_TASKS} tools** to solve completely.

## Workplace Tasks
The following O*NET workplace tasks define the scenario. Your question must be grounded in these tasks:
{TASKS}

Here are some search results about how {OCCUPATION} performs the workplace tasks to add detail and context to the scenario:

{TASK_REFERENCES}

## Available MCP Servers

{SERVER_DESCRIPTIONS}

## ⚠️ Important: Tool Execution and Validation

**You have access to the actual MCP tools listed above.** Before finalizing your response, you MUST:
1. Draft a realistic question grounded in the workplace tasks
2. Identify the specific tools and input arguments needed to answer the question
3. **Actually execute each tool** with the specified arguments to verify it works correctly
4. Ensure the tool outputs are relevant and useful for answering the question
5. Include the tool execution results in your final response

This validation step ensures the generated question can actually be solved using the available tools with the specified arguments.

## Guidelines

### Scenario Grounding
- The scenario **must be grounded in the listed workplace tasks** — they define the work being done
- The tools are the **means** to accomplish the tasks, not the starting point
- Think about how a {OCCUPATION} would naturally encounter a situation requiring all the listed workplace tasks together
- Use the search results (if available) to add realistic detail and context

### Question Realism
- Create a question that represents a real-world scenario where a {OCCUPATION} genuinely needs to perform these tasks
- The question should sound natural and authentic, as if asked by someone with a specific goal
- Include relevant context, constraints, and details that make the question engaging
- Draw on the occupation's knowledge domains and skills to make the scenario authentic

### Tool Integration and Validation
- Each tool should serve as a means to accomplish one or more of the workplace tasks
- **Execute each tool** with realistic arguments to verify it works before including it
- Consider how data flows between tools and which **dependency patterns** connect them:
  - **Parameter dependency**: One tool's output provides input for the next (e.g., a lookup result feeds into a calculation)
  - **Conditional routing**: A tool's result determines which tool to call next (e.g., an inspection finding a violation triggers a reporting tool rather than a routine filing tool)
  - **Cross-validation**: Two tools verify or contradict each other's findings on the same question
  - **Aggregation**: Parallel tool calls whose results must be combined into a single answer
- The tools should create a logical, interconnected workflow that uses one or more of these dependency patterns to address the tasks
- Consider each tool and its parent MCP server's descriptions when crafting the workflow
- Do not include exact tool names or server names in the question itself
- **Verify that tool outputs are relevant** to answering the question you've drafted

### Question Complexity
- The question should have multiple components corresponding to the workplace tasks
- Include relevant context or constraints that make multi-tool usage necessary
- Create scenarios that consist of several complementary tasks to accomplish an overall goal

### Output Format
Your response should include:
1. **Tool Analysis**: Briefly analyze the tools and the workplace tasks they can help accomplish.
2. **Cross-Tool Workflow**: Describe the workflow showing how tools will be used together, including the dependency type for each tool-to-tool link (parameter dependency, conditional routing, cross-validation, or aggregation) and any decision points where intermediate results change the next step.
3. **Target Tasks**: The specific workplace tasks and their task IDs that the question addresses.
4. **Target Tools**: The specific tools, their server names, their input arguments, AND the actual output from executing each tool. Tools should be listed in the order they would likely be called. The input arguments MUST follow the tool's Input Schema exactly — do not invent or rename parameters.
5. **Question**: A clear, realistic user question that requires tool usage to accomplish the workplace tasks.

## Output
Ensure your question is grounded in all {NUM_TASKS} workplace tasks and uses exactly {NUM_TOOLS} tools to solve completely.

**Remember to actually execute the tools and include the tool outputs in your response.**

Provide your response in the following JSON format:

```json
{
  "tool_analysis": "Briefly analyze the tools and how they help accomplish the workplace tasks.",
  "cross_tool_workflow": "Describe the workflow: for each tool-to-tool link, state the dependency type (parameter dependency, conditional routing, cross-validation, or aggregation) and note any decision points where intermediate results change the next step.",
  "target_tasks": [
    {"id": "8823", "description": "Direct or coordinate an organization's financial or budget activities to fund operations, maximize investments, or increase efficiency."},
    {"id": "8824", "description": "Confer with board members, organization officials, or staff members to discuss issues, coordinate activities, or resolve problems."}
  ],
  "target_tools": [
    {
      "server": "Server1",
      "tool": "get_weather",
      "arguments": {"location": "Paris, France"},
      "output": "Temperature: 18°C, Conditions: Partly cloudy, Humidity: 65%"
    },
    {
      "server": "Server2",
      "tool": "send_email",
      "arguments": {"to": "bob@email.com", "body": "Hi bob"},
      "output": "Email sent successfully to bob@email.com"
    }
  ],
  "question": "A clear, realistic user question grounded in the workplace tasks that requires multi-tool usage."
}
```
