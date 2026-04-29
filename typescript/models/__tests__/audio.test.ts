import { describe, it, expect, vi, beforeEach } from "vitest";
import { handleTranscription, handleTextToSpeech } from "../audio";

// Capture the latest request seen by the mocked clients so tests can assert payload shape.
const recognizeMock = vi.fn();
const synthesizeMock = vi.fn();

vi.mock("@google-cloud/speech", () => {
  const AudioEncoding = {
    ENCODING_UNSPECIFIED: 0,
    LINEAR16: 1,
    FLAC: 2,
    MULAW: 3,
    AMR: 4,
    AMR_WB: 5,
    OGG_OPUS: 6,
    SPEEX_WITH_HEADER_BYTE: 7,
    WEBM_OPUS: 9,
    MP3: 8,
  } as const;
  return {
    SpeechClient: class {
      recognize = recognizeMock;
    },
    protos: {
      google: {
        cloud: {
          speech: {
            v1: {
              RecognitionConfig: {
                AudioEncoding,
              },
            },
          },
        },
      },
    },
  };
});

vi.mock("@google-cloud/text-to-speech", () => {
  const AudioEncoding = {
    AUDIO_ENCODING_UNSPECIFIED: 0,
    LINEAR16: 1,
    MP3: 2,
    OGG_OPUS: 3,
    MULAW: 5,
    ALAW: 6,
  } as const;
  return {
    TextToSpeechClient: class {
      synthesizeSpeech = synthesizeMock;
    },
    protos: {
      google: {
        cloud: {
          texttospeech: {
            v1: {
              AudioEncoding,
            },
          },
        },
      },
    },
  };
});

const mockRuntime = {
  getSetting: vi.fn().mockImplementation((key: string) => {
    if (key === "GOOGLE_VERTEX_PROJECT_ID") return "test-project";
    if (key === "GOOGLE_VERTEX_REGION") return "us-central1";
    return null;
  }),
};

describe("Vertex Audio Handlers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    recognizeMock.mockResolvedValue([
      {
        results: [
          {
            alternatives: [{ transcript: "Hello world", confidence: 0.95 }],
          },
        ],
      },
    ]);
    synthesizeMock.mockResolvedValue([
      { audioContent: Buffer.from("mock-audio-data") },
    ]);
  });

  describe("handleTranscription", () => {
    it("transcribes a Buffer with default OGG_OPUS encoding", async () => {
      const buffer = Buffer.from("fake-audio");
      const result = await handleTranscription(mockRuntime as any, buffer);

      expect(result).toBe("Hello world");
      expect(recognizeMock).toHaveBeenCalledTimes(1);

      const req = recognizeMock.mock.calls[0]![0];
      expect(req.config.encoding).toBe(6); // OGG_OPUS
      expect(req.config.sampleRateHertz).toBe(16000);
      expect(req.config.languageCode).toBe("en-US");
      expect(req.config.model).toBe("latest_long");
      expect(req.config.useEnhanced).toBe(true);
      expect(req.config.enableAutomaticPunctuation).toBe(true);
      expect(req.audio.content).toBe(buffer.toString("base64"));
    });

    it("uses LINEAR16 encoding without sampleRateHertz for audio/wav mimeType", async () => {
      await handleTranscription(mockRuntime as any, {
        audio: Buffer.from("wav-audio"),
        mimeType: "audio/wav",
        languageCode: "fr-FR",
      });

      const req = recognizeMock.mock.calls[0]![0];
      expect(req.config.encoding).toBe(1); // LINEAR16
      expect(req.config.sampleRateHertz).toBeUndefined();
      expect(req.config.languageCode).toBe("fr-FR");
    });

    it("uses MP3 encoding for audio/mpeg mimeType", async () => {
      await handleTranscription(mockRuntime as any, {
        audio: Buffer.from("mp3-audio"),
        mimeType: "audio/mpeg",
      });

      const req = recognizeMock.mock.calls[0]![0];
      expect(req.config.encoding).toBe(8); // MP3
    });

    it("throws on unsupported mimeType", async () => {
      await expect(
        handleTranscription(mockRuntime as any, {
          audio: Buffer.from("audio"),
          mimeType: "audio/x-unknown",
        }),
      ).rejects.toThrow(/Unsupported audio mimeType/);
    });

    it("decodes base64 string input", async () => {
      const original = Buffer.from("base64-payload");
      const b64 = original.toString("base64");
      await handleTranscription(mockRuntime as any, b64);

      const req = recognizeMock.mock.calls[0]![0];
      expect(req.audio.content).toBe(b64);
    });

    it("fetches and decodes URL string input", async () => {
      const audioBytes = Buffer.from("downloaded-audio");
      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue({
        ok: true,
        headers: new Headers({ "content-length": String(audioBytes.length) }),
        body: null,
        arrayBuffer: async () =>
          audioBytes.buffer.slice(
            audioBytes.byteOffset,
            audioBytes.byteOffset + audioBytes.byteLength,
          ),
      } as unknown as Response);

      try {
        await handleTranscription(
          mockRuntime as any,
          "https://example.com/audio.ogg",
        );
        expect(fetchSpy).toHaveBeenCalledWith(
          "https://example.com/audio.ogg",
          expect.objectContaining({ signal: expect.any(AbortSignal) }),
        );
        const req = recognizeMock.mock.calls[0]![0];
        expect(req.audio.content).toBe(audioBytes.toString("base64"));
      } finally {
        fetchSpy.mockRestore();
      }
    });

    it("throws when URL fetch returns non-ok response", async () => {
      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue({
        ok: false,
        status: 404,
        statusText: "Not Found",
      } as unknown as Response);

      try {
        await expect(
          handleTranscription(
            mockRuntime as any,
            "https://example.com/missing",
          ),
        ).rejects.toThrow(/Failed to fetch audio: 404/);
      } finally {
        fetchSpy.mockRestore();
      }
    });

    it("throws when URL declares Content-Length above the cap", async () => {
      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue({
        ok: true,
        headers: new Headers({
          "content-length": String(26 * 1024 * 1024),
        }),
        body: null,
        arrayBuffer: async () => new ArrayBuffer(0),
      } as unknown as Response);

      try {
        await expect(
          handleTranscription(mockRuntime as any, "https://example.com/huge"),
        ).rejects.toThrow(/Audio file too large/);
      } finally {
        fetchSpy.mockRestore();
      }
    });

    it("accepts core TranscriptionParams shape (audioUrl)", async () => {
      const audioBytes = Buffer.from("core-shape-audio");
      const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue({
        ok: true,
        headers: new Headers({ "content-length": String(audioBytes.length) }),
        body: null,
        arrayBuffer: async () =>
          audioBytes.buffer.slice(
            audioBytes.byteOffset,
            audioBytes.byteOffset + audioBytes.byteLength,
          ),
      } as unknown as Response);

      try {
        await handleTranscription(mockRuntime as any, {
          audioUrl: "https://example.com/voice.ogg",
        });
        const req = recognizeMock.mock.calls[0]![0];
        expect(req.audio.content).toBe(audioBytes.toString("base64"));
      } finally {
        fetchSpy.mockRestore();
      }
    });

    it("rejects unrecognized object inputs with an actionable error", async () => {
      await expect(
        handleTranscription(mockRuntime as any, { foo: "bar" } as any),
      ).rejects.toThrow(/Invalid transcription input.*Buffer.*audioUrl/s);
    });
  });

  describe("handleTextToSpeech", () => {
    it("synthesizes speech from a string with OGG_OPUS audio config", async () => {
      const result = await handleTextToSpeech(mockRuntime as any, "Hello");

      expect(result).toBeInstanceOf(ArrayBuffer);
      expect(new Uint8Array(result).length).toBeGreaterThan(0);

      const req = synthesizeMock.mock.calls[0]![0];
      expect(req.input).toEqual({ text: "Hello" });
      expect(req.voice.languageCode).toBe("en-US");
      expect(req.voice.name).toBe("en-US-Neural2-F");
      expect(req.audioConfig.audioEncoding).toBe(3); // OGG_OPUS
    });

    it("uses voiceName and languageCode from local params", async () => {
      await handleTextToSpeech(mockRuntime as any, {
        text: "Bonjour",
        voiceName: "fr-FR-Wavenet-A",
        languageCode: "fr-FR",
      });

      const req = synthesizeMock.mock.calls[0]![0];
      expect(req.input.text).toBe("Bonjour");
      expect(req.voice.name).toBe("fr-FR-Wavenet-A");
      expect(req.voice.languageCode).toBe("fr-FR");
    });

    it("falls back to core 'voice' field when local 'voiceName' is absent", async () => {
      await handleTextToSpeech(mockRuntime as any, {
        text: "Hi",
        voice: "en-US-Wavenet-D",
      });

      const req = synthesizeMock.mock.calls[0]![0];
      expect(req.voice.name).toBe("en-US-Wavenet-D");
    });

    it("throws when TTS returns no audio content", async () => {
      synthesizeMock.mockResolvedValueOnce([{ audioContent: null }]);
      await expect(handleTextToSpeech(mockRuntime as any, "x")).rejects.toThrow(
        /No audio content/,
      );
    });
  });
});
