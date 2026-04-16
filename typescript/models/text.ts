import type {
  GenerateTextParams,
  IAgentRuntime,
  TextStreamResult,
} from "@elizaos/core";
import { logger, ModelType } from "@elizaos/core";
import { generateText, streamText } from "ai";
import { createVertexClient } from "../providers";
import {
  getSmallModel,
  getLargeModel,
  getReasoningSmallModel,
  getReasoningLargeModel,
} from "../utils/config";

function isOpus4Model(name: string): boolean {
  return name.toLowerCase().includes("opus-4");
}

async function generateTextWithModel(
  runtime: IAgentRuntime,
  params: GenerateTextParams,
  modelName: string,
  modelType: string,
): Promise<string | TextStreamResult> {
  const vertex = createVertexClient(runtime);

  let temperature = params.temperature ?? 0.7;
  if (isOpus4Model(modelName) && temperature !== 1) {
    temperature = 1;
  }

  const defaultMaxTokens = 8192;
  const maxTokens = Math.min(
    params.maxTokens ?? defaultMaxTokens,
    isOpus4Model(modelName) ? 32_000 : 64_000,
  );

  logger.log(`[Vertex] Using ${modelType}: ${modelName}`);

  const generateParams = {
    model: vertex(modelName),
    messages: [{ role: "user" as const, content: params.prompt }],
    system: runtime.character.system ?? undefined,
    temperature,
    maxOutputTokens: maxTokens,
    stopSequences: (params.stopSequences ?? []) as string[],
    frequencyPenalty: params.frequencyPenalty ?? 0.7,
    presencePenalty: params.presencePenalty ?? 0.7,
  };

  if (params.stream) {
    const streamResult = streamText(generateParams);
    return {
      textStream: streamResult.textStream,
      text: Promise.resolve(streamResult.text),
      usage: Promise.resolve(streamResult.usage).then((usage) => {
        if (!usage) return undefined;
        const promptTokens = usage.inputTokens ?? 0;
        const completionTokens = usage.outputTokens ?? 0;
        return {
          promptTokens,
          completionTokens,
          totalTokens: usage.totalTokens ?? promptTokens + completionTokens,
        };
      }),
      finishReason: Promise.resolve(streamResult.finishReason) as Promise<
        string | undefined
      >,
    };
  }

  const { text } = await generateText(generateParams);
  return text;
}

export async function handleTextSmall(
  runtime: IAgentRuntime,
  params: GenerateTextParams,
): Promise<string | TextStreamResult> {
  return generateTextWithModel(
    runtime,
    params,
    getSmallModel(runtime),
    ModelType.TEXT_SMALL,
  );
}

export async function handleTextLarge(
  runtime: IAgentRuntime,
  params: GenerateTextParams,
): Promise<string | TextStreamResult> {
  return generateTextWithModel(
    runtime,
    params,
    getLargeModel(runtime),
    ModelType.TEXT_LARGE,
  );
}

export async function handleReasoningSmall(
  runtime: IAgentRuntime,
  params: GenerateTextParams,
): Promise<string | TextStreamResult> {
  return generateTextWithModel(
    runtime,
    params,
    getReasoningSmallModel(runtime),
    "TEXT_REASONING_SMALL",
  );
}

export async function handleReasoningLarge(
  runtime: IAgentRuntime,
  params: GenerateTextParams,
): Promise<string | TextStreamResult> {
  return generateTextWithModel(
    runtime,
    params,
    getReasoningLargeModel(runtime),
    "TEXT_REASONING_LARGE",
  );
}
