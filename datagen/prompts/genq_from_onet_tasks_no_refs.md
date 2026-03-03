## Task
Generate a *tool-use scenario* grounded in workplace tasks performed by a given occupation.

## Objective

Brainstorm a workplace scenario in which {OCCUPATION} needs to perform *all of the following workplace tasks*, and analyze the provided MCP servers and their available tools to create
- a realistic user request that requires the use of *at least {NUM_TOOLS} target tools* to fulfill completely
- a list of target tools calls that must be made to fulfill the request and their outputs

## Workplace Tasks

**Occupation:** {OCCUPATION}
**Occupation Description:** {OCCUPATION_DESCRIPTION}
**Tasks:**
{TASKS}

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

### Output Format

Your response should include:
1. **Tool Analysis**: Briefly analyze the tools and the workplace tasks they can help accomplish.
2. **Cross-Tool Workflow**: Describe the workflow showing how tools will be used together, including the dependencies among tools and any decision points where intermediate results affect the next step.
3. **Target Tools**: The specific tools, their server names, their input arguments, AND the output from executing the tools. The input arguments MUST follow the tool's input schema exactly (including parameter names, required fields, and value types).
4. **Request**: A clear, realistic user request that requires tool usage to accomplish the workplace tasks.

## Output
Ensure your request is grounded in all {NUM_TASKS} workplace tasks and uses at least {NUM_TOOLS} tools to solve completely. Provide your response in the following JSON format:

Machine-readable schema source of truth: `prompts/genq_from_onet_tasks_output_schema.json`.

```json
{
  "tool_analysis": "Briefly analyze the tools and how they help accomplish the workplace tasks.",
  "cross_tool_workflow": "Describe the workflow: for each tool-to-tool link, state the dependency type (parameter dependency, conditional routing, cross-validation, or aggregation) and note any decision points where intermediate results change the next step.",
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
  "request": "A clear, realistic user request grounded in the workplace tasks that requires tool usage to fulfill."
}
```
