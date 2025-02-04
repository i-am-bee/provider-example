#!/usr/bin/env node

import { z } from "zod";

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

/**
 *
 * @param {McpServer} server
 */
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

/**
 *
 * @param {McpServer} server
 */
async function registerAgents(server) {
  server.agent(
    "Bee",
    "General purpose agent",
    {
      llm: z.union([z.object({ type: z.literal("ollama") })]),
      tools: z.array(z.enum("weather", "search")),
    },
    {
      prompt: z.string(),
    },
    {
      text: z.string(),
    },
    async ({ config }) => {
      const [client] = await createClientServerLinkedPair();

      const availableTools = await MCPTool.fromClient(client);
      const agent = await new BeeAgent({
        llm: createLLM(config.llm.type),
        tools: availableTools.filter((tool) =>
          config.tools.includes(tool.name)
        ),
        memory: new UnconstrainedMemory(),
      });

      return [
        async (request, { signal }) => {
          const output = await agent
            .run({ prompt: request.params.input.prompt }, { signal })
            .observe((emitter) => {
              if (request.params._meta?.progressToken)
                emitter.on("partialUpdate", ({ update: { value } }) => {
                  server.server.sendAgentRunProgress({
                    progressToken: request.params._meta.progressToken,
                    delta: {
                      text: value,
                    },
                  });
                });
            });
          return {
            text: output.result.text,
          };
        },
        () => {
          agent.destroy();
          client.close();
        },
      ];
    }
  );

  server.agent(
    "Streamlit",
    "Streamlit agent",
    {
      llm: z.union([z.object({ type: z.literal("ollama") })]),
    },
    {
      prompt: z.string(),
    },
    {
      code: z.string(),
    },
    async ({ params: { config } }) => {
      const agent = new StreamlitAgent({
        llm: createLLM(config.llm.type),
        memory: new UnconstrainedMemory(),
      });
      return [
        async ({ prompt }) => {
          const output = await agent.run({ prompt });
          return {
            code: output.result.raw,
          };
        },
        () => {
          agent.destroy();
        },
      ];
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
