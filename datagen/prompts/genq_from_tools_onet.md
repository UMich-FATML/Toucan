## Task
Generate a **Tool Use Question** based on featured MCP Servers and their tool descriptions.

## Objective
Brainstorm a compelling real-world scenario, then analyze the provided tools and their associated tasks to create a realistic user question that naturally requires the use of **all {NUM_TOOLS} tools** to solve completely.

## Guidelines

### Scenario Brainstorming
- Think of realistic, specific scenarios where someone needs to use all {NUM_TOOLS} tools to accomplish an overall goal that consists of multiple tasks
- Consider workplace tasks that {OCCUPATION} performs such as (but not limited to):
{TASKS}
- The scenario should be detailed and authentic, representing genuine use cases that consists of multiple tasks

### Question Realism
- Create questions that represent real-world scenarios where users genuinely need multiple tools
- The question should sound natural and authentic, as if asked by someone with a specific goal
- Include relevant context, constraints, and details that make the question engaging
- Consider workflows that require multiple complementary tools working together to accomplish an overall goal
- Think about how different tools support each other in real-world use cases

### Task Selection
- The question should require a sequence or combination of tool calls to solve completely
- Choose tools based on how they complement each other to accomplish an overall goal
- Consider each tool and its parent MCP server's descriptions when crafting the cross-tool workflow
- Ensure the tools create a logical, interconnected workflow

### Question Complexity
- Create questions that are complex enough to warrant using all {NUM_TOOLS} tools
- The question should have multiple components or require several steps that perform different tasks to accomplish an overall goal
- Include relevant context or constraints that make the multi-server tool usage necessary
- Do not contain the exact tool names or server names in the question
- Create scenarios that consist of several complementary tasks to accomplish an overall goal

### Cross-Tool Integration
- Think about how different tools' capabilities can be combined
- Consider data flow between different tasks (eg, pass the output from one task as input to another task)
- Create realistic scenarios that consist of multiple complementary tasks to accomplish an overall goal
- Focus on complementary functionalities across different tools

### Output Format
Your response should include:
1. **Tool Analysis**: Briefly analyze the tools and their associated tasks, focusing on how they work together to accomplish an overall goal.
2. **Cross-Tool Workflow**: Describe the workflow showing how tools will be used together to accomplish an overall goal.
3. **Target Tasks**: The specific tasks and their task IDs that must be completed to solve the question.
4. **Target Tools**: The specific tools and their server names that must be used to solve the question, in the order they would likely be called.
5. **Question**: A clear, realistic user question that requires tool usage.

## Available Tools

{TOOL_DESCRIPTIONS}

## Output
Ensure your question requires all {NUM_TOOLS} tools to solve completely. Provide your response in the following XML format:

<response>
  <tool_analysis>
    <!-- Briefly analyze the tools and their associated tasks, focusing on how they work together to accomplish an overall goal. -->
  </tool_analysis>
  <cross_tool_workflow>
    <!-- Describe the workflow showing how tools will be used together to solve the question. -->
  </cross_tool_workflow>
  <target_tasks>
    <!-- The specific tasks and their task IDs that must be completed to solve the question. e.g., <task id="8823">Direct or coordinate an organization's financial or budget activities to fund operations, maximize investments, or increase efficiency.</task> <task id="8824">Confer with board members, organization officials, or staff members to discuss issues, coordinate activities, or resolve problems.</task> -->
  </target_tasks>
  <target_tools>
    <!-- The specific tools and their server names that should be used together, listed in order with their server names. e.g., <tool server="Server1">search_posts</tool> <tool server="Server2">send_email</tool> -->
  </target_tools>
  <question>
    <!-- A clear, realistic user question that requires multi-server tool usage spanning different services/domains. -->
  </question>
</response> 