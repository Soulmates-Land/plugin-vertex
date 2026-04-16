import { createVertexAnthropic } from "@ai-sdk/google-vertex/anthropic";
import type { IAgentRuntime } from "@elizaos/core";
import { logger } from "@elizaos/core";

export function createVertexClient(runtime: IAgentRuntime) {
  const projectId =
    String(runtime.getSetting("GOOGLE_VERTEX_PROJECT_ID") ?? "") ||
    process.env.GOOGLE_VERTEX_PROJECT_ID;
  const region =
    String(runtime.getSetting("GOOGLE_VERTEX_REGION") ?? "") ||
    process.env.GOOGLE_VERTEX_REGION ||
    "us-east5";

  if (!projectId) {
    throw new Error(
      "GOOGLE_VERTEX_PROJECT_ID is required for the Vertex AI plugin",
    );
  }

  logger.debug(
    `[Vertex] Creating client: project=${projectId} region=${region}`,
  );

  return createVertexAnthropic({ project: projectId, location: region });
}
