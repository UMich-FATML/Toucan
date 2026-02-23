You have access to the '{SERVER_NAME}' MCP server. Your job is to test ONE specific tool and report the quality of its output.

Tool to test:
{TOOL_ENTRY}

Instructions:
1. Call this tool at least once using the most realistic inputs you can construct based on its description and input schema
2. If the tool returns an error caused by your inputs (e.g. invalid argument value, missing required field), retry with corrected inputs (up to 2 more attempts)
3. If the tool fails due to a server or connection error (e.g. 500, timeout, unreachable), note the failure and stop
4. Output ONLY a JSON object as your final message — no markdown fences, no explanation, just the raw JSON:

{
  "tool_results": [
    {
      "tool_name": "<exact tool name>",
      "quality": "<pass or fail>",
      "reasoning": "<brief explanation of what the tool returned and why it passes or fails>"
    }
  ]
}

"quality" must be "pass" if the tool returned real, non-error data consistent with its described purpose.
"quality" must be "fail" if the tool only returned errors on all attempts, was unreachable, or returned empty/meaningless output.
