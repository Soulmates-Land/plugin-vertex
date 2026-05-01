import type {
  IAgentRuntime,
  TranscriptionParams as CoreTranscriptionParams,
  TextToSpeechParams as CoreTextToSpeechParams,
} from "@elizaos/core";
import { logger } from "@elizaos/core";
import { v2 as speechV2, protos as speechProtos } from "@google-cloud/speech";
import {
  TextToSpeechClient,
  protos as ttsProtos,
} from "@google-cloud/text-to-speech";
import {
  getProjectId,
  getRegion,
  getTranscriptionModel,
  getTranscriptionRegion,
  getTTSModel,
} from "../utils/config";

// Types for local parameters (extra fields not in @elizaos/core's TranscriptionParams).
// `audio` is intentionally Buffer-only here; callers with Blob/string should pass
// a string at the top level (URL or base64) instead.
export interface LocalTranscriptionParams {
  audio: Buffer;
  languageCode?: string;
  mimeType?: string;
}

export interface LocalTextToSpeechParams {
  text: string;
  voiceName?: string;
  languageCode?: string;
}

type TranscriptionInput =
  | LocalTranscriptionParams
  | CoreTranscriptionParams
  | Buffer
  | string;
type TTSInput = LocalTextToSpeechParams | CoreTextToSpeechParams | string;

type IRecognizeRequest = speechProtos.google.cloud.speech.v2.IRecognizeRequest;
type ISynthesizeSpeechRequest =
  ttsProtos.google.cloud.texttospeech.v1.ISynthesizeSpeechRequest;
const TtsAudioEncoding = ttsProtos.google.cloud.texttospeech.v1.AudioEncoding;

// URL fetch limits
const URL_FETCH_TIMEOUT_MS = 30_000;
const URL_FETCH_MAX_BYTES = 25 * 1024 * 1024; // 25 MB

// Module-level client cache, keyed by apiEndpoint (region-specific).
const speechClientCache = new Map<string, speechV2.SpeechClient>();
const ttsClientCache = new Map<string, TextToSpeechClient>();

function getSpeechClient(
  apiEndpoint?: string,
  projectId?: string,
): speechV2.SpeechClient {
  const key = `${apiEndpoint ?? "default"}:${projectId ?? "default"}`;
  let client = speechClientCache.get(key);
  if (!client) {
    client = new speechV2.SpeechClient({
      ...(apiEndpoint ? { apiEndpoint } : {}),
      ...(projectId ? { projectId, quotaProject: projectId } : {}),
    });
    speechClientCache.set(key, client);
  }
  return client;
}

function getTextToSpeechClient(
  apiEndpoint?: string,
  projectId?: string,
): TextToSpeechClient {
  const key = `${apiEndpoint ?? "default"}:${projectId ?? "default"}`;
  let client = ttsClientCache.get(key);
  if (!client) {
    client = new TextToSpeechClient({
      ...(apiEndpoint ? { apiEndpoint } : {}),
      ...(projectId ? { projectId, quotaProject: projectId } : {}),
    });
    ttsClientCache.set(key, client);
  }
  return client;
}



/**
 * Fetches an audio URL with a hard timeout, response.ok check, and a size cap.
 * Restricted to https:// to mitigate SSRF — internal endpoints (169.254.169.254,
 * localhost, link-local) are commonly served over http://.
 */
async function fetchAudioUrl(url: string): Promise<Buffer> {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error(`Invalid audio URL: ${url}`);
  }
  if (parsed.protocol !== "https:") {
    throw new Error(
      `Refusing to fetch audio from non-https URL (got ${parsed.protocol}).`,
    );
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), URL_FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      throw new Error(
        `Failed to fetch audio: ${response.status} ${response.statusText}`,
      );
    }
    const contentLength = response.headers.get("content-length");
    if (contentLength) {
      const declared = Number.parseInt(contentLength, 10);
      if (Number.isFinite(declared) && declared > URL_FETCH_MAX_BYTES) {
        throw new Error(
          `Audio file too large: ${declared} bytes exceeds ${URL_FETCH_MAX_BYTES} byte limit.`,
        );
      }
    }
    if (!response.body) {
      const ab = await response.arrayBuffer();
      if (ab.byteLength > URL_FETCH_MAX_BYTES) {
        throw new Error(
          `Audio file too large: ${ab.byteLength} bytes exceeds ${URL_FETCH_MAX_BYTES} byte limit.`,
        );
      }
      return Buffer.from(ab);
    }
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value) {
        total += value.byteLength;
        if (total > URL_FETCH_MAX_BYTES) {
          await reader.cancel();
          throw new Error(
            `Audio file too large: exceeded ${URL_FETCH_MAX_BYTES} byte limit while streaming.`,
          );
        }
        chunks.push(value);
      }
    }
    return Buffer.concat(
      chunks.map((c) => Buffer.from(c)),
      total,
    );
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Handles audio transcription using Google Cloud Speech-to-Text.
 * Supports Buffer, URL string, base64 string, or parameter objects.
 *
 * @param runtime - The agent runtime for accessing settings and providers.
 * @param input - The audio input to transcribe (Buffer, URL/base64 string, or TranscriptionParams).
 * @returns The transcribed text.
 */
export async function handleTranscription(
  runtime: IAgentRuntime,
  input: TranscriptionInput,
): Promise<string> {
  const region = getTranscriptionRegion(runtime);
  const projectId = getProjectId(runtime);
  const apiEndpoint =
    !region || region === "global" ? undefined : `${region}-speech.googleapis.com`;
  const client = getSpeechClient(apiEndpoint, projectId);

  let audioContent: Buffer;
  const model = getTranscriptionModel(runtime);
  let languageCode = "en-US";

  if (Buffer.isBuffer(input)) {
    audioContent = input;
  } else if (typeof input === "string") {
    // String inputs: HTTP(S) URL is fetched, otherwise treated as base64.
    if (/^https?:\/\//i.test(input)) {
      audioContent = await fetchAudioUrl(input);
    } else {
      audioContent = Buffer.from(input, "base64");
    }
  } else if ("audio" in input && Buffer.isBuffer(input.audio)) {
    audioContent = input.audio;
    languageCode = input.languageCode ?? languageCode;
  } else if ("audioUrl" in input && typeof input.audioUrl === "string") {
    audioContent = await fetchAudioUrl(input.audioUrl);
  } else {
    throw new Error(
      "Invalid transcription input. Accepted: Buffer, URL string, base64 string, " +
        "{ audio: Buffer, languageCode?, mimeType? }, or { audioUrl: string }.",
    );
  }

  let actualProjectId = projectId;
  if (!actualProjectId) {
    actualProjectId = await client.getProjectId();
  }
  const location = region && region !== "global" ? region : "global";
  const recognizer = `projects/${actualProjectId}/locations/${location}/recognizers/_`;

  logger.log(
    `[Vertex] Transcribing audio with model ${model} (V2 API, auto-decoding)...`,
  );

  const request: IRecognizeRequest = {
    recognizer,
    config: {
      autoDecodingConfig: {},
      languageCodes: [languageCode],
      model,
      features: {
        enableAutomaticPunctuation: true,
      },
    },
    content: audioContent.toString("base64"),
  };

  const [response] = await client.recognize(request);
  const transcription = response.results
    ?.map((result) => result.alternatives?.[0]?.transcript)
    .join("\n");

  const confidence = response.results?.[0]?.alternatives?.[0]?.confidence ?? 0;
  logger.log(`[Vertex] Transcription completed. Confidence: ${confidence}`);

  return transcription ?? "";
}

/**
 * Handles text-to-speech synthesis using Google Cloud Text-to-Speech.
 *
 * @param runtime - The agent runtime for accessing settings and providers.
 * @param input - The text input to synthesize (string, TextToSpeechParams, or local params).
 * @returns An ArrayBuffer containing the synthesized audio in OGG_OPUS format.
 */
export async function handleTextToSpeech(
  runtime: IAgentRuntime,
  input: TTSInput,
): Promise<ArrayBuffer> {
  const region = getRegion(runtime);
  const projectId = getProjectId(runtime);
  const apiEndpoint =
    !region || region === "global"
      ? undefined
      : `${region}-texttospeech.googleapis.com`;
  const client = getTextToSpeechClient(apiEndpoint, projectId);

  let text = "";
  let voiceName = getTTSModel(runtime); // Default neural voice
  let languageCode = "en-US";

  if (typeof input === "string") {
    text = input;
  } else if ("text" in input) {
    text = input.text;
    if ("voiceName" in input && input.voiceName) {
      voiceName = input.voiceName;
    } else if ("voice" in input && input.voice) {
      voiceName = input.voice;
    }
    if ("languageCode" in input && input.languageCode) {
      languageCode = input.languageCode;
    }
  } else {
    throw new Error(
      "Invalid TTS input. Accepted: string or { text: string, voice?, voiceName?, languageCode? }.",
    );
  }
  if (text.trim().length === 0) {
    throw new Error("TTS input text is empty.");
  }

  logger.log(
    `[Vertex] Synthesizing speech for text: "${text.substring(0, 50)}..."`,
  );

  const request: ISynthesizeSpeechRequest = {
    input: { text },
    voice: { languageCode, name: voiceName },
    audioConfig: { audioEncoding: TtsAudioEncoding.OGG_OPUS },
  };

  const [response] = await client.synthesizeSpeech(request);
  if (!response.audioContent) {
    throw new Error("No audio content received from TTS");
  }

  // Convert Buffer (or Uint8Array/string) to ArrayBuffer for Eliza core.
  const audio = response.audioContent;
  let buffer: Buffer;
  if (Buffer.isBuffer(audio)) {
    buffer = audio;
  } else if (typeof audio === "string") {
    buffer = Buffer.from(audio, "base64");
  } else {
    buffer = Buffer.from(audio);
  }
  const arrayBuffer = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength,
  );

  logger.log(
    `[Vertex] Speech synthesis completed. Audio size: ${buffer.length} bytes`,
  );
  return arrayBuffer as ArrayBuffer;
}
