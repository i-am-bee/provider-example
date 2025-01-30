#!/usr/bin/env node

import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

import { OllamaChatLLM } from "bee-agent-framework/adapters/ollama/chat";
import { BeeAgent } from "bee-agent-framework/agents/bee/agent";
import { StreamlitAgent } from "bee-agent-framework/agents/experimental/streamlit/agent";
import { UnconstrainedMemory } from "bee-agent-framework/memory/unconstrainedMemory";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { OpenMeteoTool } from "bee-agent-framework/tools/weather/openMeteo";
import { DuckDuckGoSearchTool } from "bee-agent-framework/tools/search/duckDuckGoSearch";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { MCPTool } from "bee-agent-framework/tools/mcp";

function createLLM(type) {
  switch (type) {
    case "ollama":
      return new OllamaChatLLM();
    default:
      throw new Error(`Unsupported llm ${type}`);
  }
}

async function registerTools(server) {
  const weatherTool = new OpenMeteoTool();
  server.tool(
    "weather",
    weatherTool.description,
    { ...(await weatherTool.inputSchema().shape) },
    (args) => {
      return weatherTool.run(args);
    }
  );

  const searchTool = new DuckDuckGoSearchTool();
  server.tool(
    "search",
    searchTool.description,
    { ...(await weatherTool.inputSchema().shape) },
    (args) => {
      return searchTool.run(args);
    }
  );
}

async function registerAgents(server) {
  server.agent(
    "Bee",
    "General purpose agent",
    {
      llm: z.union([z.object({ type: z.literal("ollama") })]),
      tools: z.array(z.enum("weather", "search")),
    },
    async ({ config, prompt }) => {
      const [client] = await createClientServerLinkedPair();
      try {
        const availableTools = await MCPTool.fromClient(client);
        const output = await new BeeAgent({
          llm: createLLM(config.llm.type),
          tools: availableTools.filter((tool) =>
            config.tools.includes(tool.name)
          ),
          memory: new UnconstrainedMemory(),
        }).run({
          prompt,
        });
        return {
          text: output.result.text,
        };
      } finally {
        await client.close();
      }
    }
  );

  server.agent(
    "Streamlit",
    "Streamlit agent",
    {
      llm: z.union([z.object({ type: z.literal("ollama") })]),
      tools: z.array(z.enum("weather", "search")),
    },
    async ({ config, prompt }) => {
      const output = await new StreamlitAgent({
        llm: createLLM(config.llm.type),
        memory: new UnconstrainedMemory(),
      }).run({ prompt });
      return {
        text: output.result.raw,
      };
    }
  );
}

async function createClientServerLinkedPair() {
  const client = new Client(
    { name: "memory", version: "0.0.0" },
    { capabilities: {} }
  );
  const server = await createServer({ tools: true });
  const [clientTransport, serverTransport] =
    InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  await client.connect(clientTransport);
  return [client, server];
}

export async function createServer(
  { tools, agents } = { tools: true, agents: true }
) {
  const server = new McpServer(
    {
      name: "BeeAI Agents",
      version: "1.0.0",
    },
    {
      capabilities: {
        tools: tools ? {} : undefined,
        agents: agents ? {} : undefined,
      },
    }
  );

  if (tools) await registerTools(server);
  if (agents) await registerAgents(server);

  return server;
}

export async function runServer() {
  const server = await createServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

await runServer();
