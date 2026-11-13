// Timestamp a supplied audio file using already-installed macOS 26 speech assets.
// No microphone, network transcription, or model installation is used.
import Foundation
import Speech
import AVFoundation
import CoreMedia

struct TimedWord: Codable {
    let text: String
    let start: Double
    let end: Double
}

struct TimedSegment: Codable {
    let text: String
    let start: Double
    let end: Double
    let words: [TimedWord]
}

guard CommandLine.arguments.count == 3 else {
    fputs("Usage: swift transcribe.swift input.wav output.json\n", stderr)
    exit(2)
}
let inputPath = CommandLine.arguments[1]
let outputPath = CommandLine.arguments[2]

Task {
    do {
        let locale = Locale(identifier: "en_US")
        let installed = await SpeechTranscriber.installedLocales
        guard SpeechTranscriber.isAvailable,
              installed.contains(where: { $0.identifier == locale.identifier }) else {
            throw NSError(domain: "AlphabetTranscription", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                            "English speech assets are not installed; no download attempted."])
        }
        let transcriber = SpeechTranscriber(
            locale: locale, transcriptionOptions: [], reportingOptions: [],
            attributeOptions: [.audioTimeRange, .transcriptionConfidence]
        )
        let collector = Task { () throws -> [TimedSegment] in
            var segments: [TimedSegment] = []
            for try await result in transcriber.results {
                var words: [TimedWord] = []
                for run in result.text.runs {
                    if let time = run.audioTimeRange {
                        words.append(TimedWord(
                            text: String(result.text[run.range].characters),
                            start: time.start.seconds,
                            end: CMTimeRangeGetEnd(time).seconds
                        ))
                    }
                }
                segments.append(TimedSegment(
                    text: String(result.text.characters),
                    start: result.range.start.seconds,
                    end: CMTimeRangeGetEnd(result.range).seconds,
                    words: words
                ))
            }
            return segments
        }
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: inputPath))
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        try await analyzer.start(inputAudioFile: file, finishAfterFile: true)
        let segments = try await collector.value
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(segments).write(to: URL(fileURLWithPath: outputPath))
        print("Wrote \(segments.count) timestamped speech segments to \(outputPath)")
        exit(0)
    } catch {
        fputs("Transcription failed: \(error)\n", stderr)
        exit(1)
    }
}
dispatchMain()
