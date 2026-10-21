import "dotenv/config";
import { createDeepAgent } from "deepagents";
import { ChatGroq } from "@langchain/groq";
import { tool, todoListMiddleware } from "langchain";
import { TavilySearch } from "@langchain/tavily";
import { z } from "zod";

const model = new ChatGroq({model: "openai/gpt-oss-120b",temperature: 0,});

const webSearch = tool(
  async ({
    query,
    maxResults = 5,
    topic = "general",
  }) => {
    const search = new TavilySearch({
      tavilyApiKey: process.env.TAVILY_API_KEY,
      maxResults,
      topic,
    });

    return await search._call({ query });
  },
  {
    name: "web_search",
    description:"Search the internet for current information and research sources.",
    schema: z.object({
      query: z.string().describe("What to search for"),
      maxResults: z.number().optional().default(5),
      topic: z
        .enum(["general", "news", "finance"])
        .optional()
        .default("general"),
    }),
  }
);

const agent = await createDeepAgent({
  model,
  tools: [webSearch],
  middleware: [todoListMiddleware()],
  systemPrompt: `
    You are an expert research assistant.
    For complex tasks:
    1. Break the task into smaller steps.
    2. Use web_search to collect relevant information.
    3. Compare information from multiple sources.
    4. Delegate subtasks to subagents when useful.
    5. Use filesystem tools to organize research if needed.
    6. Provide a clear, accurate final answer.
    7. Do not invent facts or sources.
  `,
});

const result = await agent.invoke({
  messages: [
    {
      role: "user",
      content:
        "Research the latest AI agent frameworks and compare LangChain, LangGraph, and Deep Agents.",
    },
  ],
});

const lastMessage = result.messages.at(-1);

console.log("\n--- Deep Agent Result ---\n");
console.log(lastMessage.content);
