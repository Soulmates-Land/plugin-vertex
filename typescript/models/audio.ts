import type {
  IAgentRuntime,
  TranscriptionParams as CoreTranscriptionParams,
  TextToSpeechParams as CoreTextToSpeechParams,
} from "@elizaos/core";
import { logger } from "@elizaos/core";
import { SpeechClient } from "@google-cloud/speech";
import { TextToSpeechClient } from "@google-cloud/text-to-speech";
import { getProjectId, getRegion, getTranscriptionModel, getTTSModel } from "../utils/config";

// Types for local parameters
export interface LocalTranscriptionParams {
    audio: Buffer | Blob | string;
    model?: string;
    languageCode?: string;
    mimeType?: string;
}

export interface LocalTextToSpeechParams {
    text: string;
    model?: string;
    voiceName?: string;
    languageCode?: string;
}

type TranscriptionInput = LocalTranscriptionParams | CoreTranscriptionParams | Buffer | string;
type TTSInput = LocalTextToSpeechParams | CoreTextToSpeechParams | string;

/**
 * Handles audio transcription using Google Cloud Speech-to-Text.
 * Supports Buffer, URL string, or parameter objects.
 * 
 * @param runtime - The agent runtime for accessing settings and providers.
 * @param input - The audio input to transcribe (Buffer, URL, or TranscriptionParams).
 * @returns The transcribed text.
 * @throws Error if the GCP project ID is not set or if transcription fails.
 */
export async function handleTranscription(
  runtime: IAgentRuntime,
  input: TranscriptionInput
): Promise<string> {
  const projectId = getProjectId(runtime);
  if (!projectId) throw new Error("GOOGLE_VERTEX_PROJECT_ID not set");

  const region = getRegion(runtime);
  const client = new SpeechClient({
    apiEndpoint: `${region}-speech.googleapis.com`,
  });
  let audioContent: Buffer;
  let model = getTranscriptionModel(runtime);
  let languageCode = "en-US";
  let encoding: any = "OGG_OPUS"; // Default for WhatsApp
  let sampleRateHertz = 16000;

  if (Buffer.isBuffer(input)) {
    audioContent = input;
  } else if (typeof input === "string") {
    // Assume it's a URL or base64? For simplicity, we'll try to fetch if it looks like a URL
    if (input.startsWith("http")) {
        const response = await fetch(input);
        const arrayBuffer = await response.arrayBuffer();
        audioContent = Buffer.from(arrayBuffer);
    } else {
        // Assume base64
        audioContent = Buffer.from(input, "base64");
    }
  } else if ("audio" in input && Buffer.isBuffer(input.audio)) {
    audioContent = input.audio;
    model = (input as LocalTranscriptionParams).model ?? model;
    languageCode = (input as LocalTranscriptionParams).languageCode ?? languageCode;
  } else if ("audioUrl" in input && typeof (input as any).audioUrl === "string") {
    const response = await fetch((input as any).audioUrl);
    const arrayBuffer = await response.arrayBuffer();
    audioContent = Buffer.from(arrayBuffer);
    languageCode = (input as any).languageCode ?? languageCode;
  } else {
    throw new Error("Invalid transcription input");
  }

  logger.log(`[Vertex] Transcribing audio with model ${model}...`);

  const request = {
    audio: {
      content: audioContent.toString("base64"),
    },
    config: {
      encoding: encoding,
      sampleRateHertz: sampleRateHertz,
      languageCode: languageCode,
      model: model,
      useEnhanced: true,
      enableAutomaticPunctuation: true,
    },
  };

  try {
    const [response] = await client.recognize(request as any);
    const transcription = response.results
      ?.map(result => result.alternatives?.[0]?.transcript)
      .join("\n");

    const confidence = response.results?.[0]?.alternatives?.[0]?.confidence ?? 0;
    logger.log(`[Vertex] Transcription completed. Confidence: ${confidence}`);

    return transcription ?? "";
  } catch (error: any) {
    logger.error(`[Vertex] Transcription error: ${error.message}`);
    throw error;
  }
}

/**
 * Handles text-to-speech synthesis using Google Cloud Text-to-Speech.
 * 
 * @param runtime - The agent runtime for accessing settings and providers.
 * @param input - The text input to synthesize (string or TextToSpeechParams).
 * @returns An ArrayBuffer containing the synthesized audio in OGG_OPUS format.
 * @throws Error if the GCP project ID is not set or if synthesis fails.
 */
export async function handleTextToSpeech(
  runtime: IAgentRuntime,
  input: TTSInput
): Promise<ArrayBuffer> {
  const projectId = getProjectId(runtime);
  if (!projectId) throw new Error("GOOGLE_VERTEX_PROJECT_ID not set");

  const client = new TextToSpeechClient();
  let text = "";
  let voiceName = getTTSModel(runtime); // Using default neural voice
  let languageCode = "en-US";

  if (typeof input === "string") {
    text = input;
  } else if ("text" in input) {
    text = input.text;
    voiceName = (input as LocalTextToSpeechParams).voiceName ?? voiceName;
    languageCode = (input as any).languageCode ?? languageCode;
  }

  logger.log(`[Vertex] Synthesizing speech for text: "${text.substring(0, 50)}..."`);

  const request = {
    input: { text },
    voice: { languageCode, name: voiceName },
    audioConfig: { audioEncoding: "OGG_OPUS" as const },
  };

  try {
    const [response] = await client.synthesizeSpeech(request);
    if (!response.audioContent) {
      throw new Error("No audio content received from TTS");
    }
    
    // Convert Buffer to ArrayBuffer as expected by Eliza core
    const buffer = response.audioContent as Buffer;
    const arrayBuffer = buffer.buffer.slice(
        buffer.byteOffset,
        buffer.byteOffset + buffer.byteLength
    );

    logger.log(`[Vertex] Speech synthesis completed. Audio size: ${buffer.length} bytes`);
    return arrayBuffer as ArrayBuffer;
  } catch (error: any) {
    logger.error(`[Vertex] TTS error: ${error.message}`);
    throw error;
  }
}
