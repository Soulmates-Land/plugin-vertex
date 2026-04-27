import { describe, it, expect, vi, beforeEach } from "vitest";
import { handleTranscription, handleTextToSpeech } from "../audio";
import { SpeechClient } from "@google-cloud/speech";
import { TextToSpeechClient } from "@google-cloud/text-to-speech";

// Mock the Google Cloud clients
vi.mock("@google-cloud/speech", () => {
    return {
        SpeechClient: class {
            recognize = vi.fn().mockResolvedValue([{
                results: [{
                    alternatives: [{
                        transcript: "Hello world",
                        confidence: 0.95
                    }]
                }]
            }]);
        }
    };
});

vi.mock("@google-cloud/text-to-speech", () => {
    return {
        TextToSpeechClient: class {
            synthesizeSpeech = vi.fn().mockResolvedValue([{
                audioContent: Buffer.from("mock-audio-data")
            }]);
        }
    };
});

// Mock runtime
const mockRuntime = {
    getSetting: vi.fn().mockImplementation((key: string) => {
        if (key === "GOOGLE_VERTEX_PROJECT_ID") return "test-project";
        return null;
    }),
};

describe("Vertex Audio Handlers", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe("handleTranscription", () => {
        it("should transcribe a Buffer successfully", async () => {
            const buffer = Buffer.from("fake-audio");
            const result = await handleTranscription(mockRuntime as any, buffer);
            expect(result).toBe("Hello world");
        });

        it("should throw error if project ID is missing", async () => {
            const runtimeWithoutProject = { getSetting: vi.fn().mockReturnValue(null) };
            await expect(handleTranscription(runtimeWithoutProject as any, Buffer.from("")))
                .rejects.toThrow("GOOGLE_VERTEX_PROJECT_ID not set");
        });
    });

    describe("handleTextToSpeech", () => {
        it("should synthesize speech successfully", async () => {
            const result = await handleTextToSpeech(mockRuntime as any, "Hello");
            expect(result).toBeInstanceOf(ArrayBuffer);
            expect(new Uint8Array(result).length).toBeGreaterThan(0);
        });
    });
});
