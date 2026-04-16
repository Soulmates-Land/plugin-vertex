import type { IAgentRuntime, ObjectGenerationParams } from "@elizaos/core";
import { logger, ModelType } from "@elizaos/core";
import { generateObject, jsonSchema } from "ai";
import { createVertexClient } from "../providers";
import { getSmallModel, getLargeModel } from "../utils/config";

async function generateObjectWithModel(
  runtime: IAgentRuntime,
  params: ObjectGenerationParams,
  modelName: string,
  modelType: string,
): Promise<Record<string, unknown>> {
  const vertex = createVertexClient(runtime);

  logger.log(`[Vertex] Object generation using ${modelType}: ${modelName}`);

  const { object } = await generateObject({
    model: vertex(modelName),
    messages: [{ role: "user" as const, content: params.prompt }],
    system: runtime.character.system ?? undefined,
    schema: jsonSchema(params.schema ?? { type: "object" }),
    temperature: params.temperature ?? 0.7,
    maxOutputTokens: params.maxTokens ?? 8192,
  });

  return object as Record<string, unknown>;
}

export async function handleObjectSmall(
  runtime: IAgentRuntime,
  params: ObjectGenerationParams,
): Promise<Record<string, unknown>> {
  return generateObjectWithModel(
    runtime,
    params,
    getSmallModel(runtime),
    ModelType.OBJECT_SMALL,
  );
}

export async function handleObjectLarge(
  runtime: IAgentRuntime,
  params: ObjectGenerationParams,
): Promise<Record<string, unknown>> {
  return generateObjectWithModel(
    runtime,
    params,
    getLargeModel(runtime),
    ModelType.OBJECT_LARGE,
  );
}
