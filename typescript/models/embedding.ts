import type { IAgentRuntime, TextEmbeddingParams } from "@elizaos/core";
import { logger } from "@elizaos/core";
import { embed } from "ai";
import { createVertex } from "@ai-sdk/google-vertex";
import { executeWithRetry } from "../utils/retry";

const DEFAULT_EMBEDDING_MODEL = "text-embedding-005";

function getEmbeddingModel(runtime: IAgentRuntime): string {
  const setting = runtime.getSetting("VERTEX_EMBEDDING_MODEL");
  if (typeof setting === "string" && setting.length > 0) return setting;
  return process.env.VERTEX_EMBEDDING_MODEL ?? DEFAULT_EMBEDDING_MODEL;
}

function createGoogleVertexClient(runtime: IAgentRuntime) {
  const projectId =
    String(runtime.getSetting("GOOGLE_VERTEX_PROJECT_ID") ?? "") ||
    process.env.GOOGLE_VERTEX_PROJECT_ID;
  const region =
    String(runtime.getSetting("GOOGLE_VERTEX_REGION") ?? "") ||
    process.env.GOOGLE_VERTEX_REGION ||
    "us-east5";

  if (!projectId) {
    throw new Error("GOOGLE_VERTEX_PROJECT_ID is required for embeddings");
  }

  return createVertex({ project: projectId, location: region });
}

export async function handleTextEmbedding(
  runtime: IAgentRuntime,
  params: TextEmbeddingParams | string | null,
): Promise<number[]> {
  const text = typeof params === "string" ? params : (params?.text ?? "");

  if (!text) return [];

  const modelName = getEmbeddingModel(runtime);
  const vertex = createGoogleVertexClient(runtime);

  logger.debug(`[Vertex] Embedding using ${modelName}`);

  const dimensionSetting = runtime.getSetting("VERTEX_EMBEDDING_DIMENSIONS");
  const outputDimensionality =
    typeof dimensionSetting === "number"
      ? dimensionSetting
      : typeof dimensionSetting === "string"
        ? parseInt(dimensionSetting, 10)
        : undefined;

  const { embedding } = await executeWithRetry("embedding request", () =>
    embed({
      model: vertex.textEmbeddingModel(modelName),
      value: text,
      ...(outputDimensionality
        ? {
            providerOptions: {
              google: { outputDimensionality },
            },
          }
        : {}),
    }),
  );

  return embedding;
}
