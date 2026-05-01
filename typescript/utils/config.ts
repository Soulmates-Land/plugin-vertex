import type { IAgentRuntime } from "@elizaos/core";

const DEFAULT_SMALL_MODEL = "claude-haiku-4-5";
const DEFAULT_LARGE_MODEL = "claude-sonnet-4-6";
const DEFAULT_REASONING_SMALL_MODEL = "claude-sonnet-4-6";
const DEFAULT_REASONING_LARGE_MODEL = "claude-opus-4-6";

function getSetting(runtime: IAgentRuntime, key: string): string | undefined {
  const value = runtime.getSetting(key);
  if (typeof value === "string" && value.length > 0) return value;
  const env = process.env[key];
  if (typeof env === "string" && env.length > 0) return env;
  return undefined;
}

export function getProjectId(runtime: IAgentRuntime): string | undefined {
  return getSetting(runtime, "GOOGLE_VERTEX_PROJECT_ID");
}

export function getSmallModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, "VERTEX_SMALL_MODEL") ??
    getSetting(runtime, "SMALL_MODEL") ??
    DEFAULT_SMALL_MODEL
  );
}

export function getLargeModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, "VERTEX_LARGE_MODEL") ??
    getSetting(runtime, "LARGE_MODEL") ??
    DEFAULT_LARGE_MODEL
  );
}

export function getReasoningSmallModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, "VERTEX_REASONING_SMALL_MODEL") ??
    DEFAULT_REASONING_SMALL_MODEL
  );
}

export function getReasoningLargeModel(runtime: IAgentRuntime): string {
  return (
    getSetting(runtime, "VERTEX_REASONING_LARGE_MODEL") ??
    DEFAULT_REASONING_LARGE_MODEL
  );
}

// Default to "chirp": the Universal Speech Model, recommended for voice notes and conversational audio.
// Note: chirp and chirp_2 are only available via the Speech v2 API.
// Override with VERTEX_TRANSCRIPTION_MODEL if a different v2 model is desired (e.g. "latest_long").
export function getTranscriptionModel(runtime: IAgentRuntime): string {
  return getSetting(runtime, "VERTEX_TRANSCRIPTION_MODEL") ?? "chirp";
}

export function getTTSModel(runtime: IAgentRuntime): string {
  return getSetting(runtime, "VERTEX_TTS_MODEL") ?? "en-US-Chirp3-HD-Achernar";
}

export function getRegion(runtime: IAgentRuntime): string {
  return getSetting(runtime, "GOOGLE_VERTEX_REGION") ?? "us-central1";
}

export function getTranscriptionRegion(runtime: IAgentRuntime): string {
  return getSetting(runtime, "GOOGLE_SPEECH_REGION") ?? "us-central1";
}
