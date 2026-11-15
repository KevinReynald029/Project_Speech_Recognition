import numpy as np
import soundfile as sf

def add_white_noise(audio, snr_db):
    signal_power = np.mean(audio ** 2)

    if signal_power == 0:
        return audio.copy(), None

    noise = np.random.normal(0, 1, len(audio))
    noise_power = np.mean(noise ** 2)

    target_noise_power = (signal_power / (10 ** (snr_db / 10)))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    noisy_audio = audio + noise
    measured_snr = 10 * np.log10(signal_power / np.mean(noise ** 2))
    peak = np.max(np.abs(noisy_audio))

    if peak > 0.99:
        scale = 0.99 / peak
        noisy_audio = noisy_audio * scale

    return noisy_audio, measured_snr

if __name__ == "__main__":
    audio_path = "examples/real_sample.wav"
    audio, sample_rate = sf.read(audio_path)

    for snr_db in [20, 10, 5]:
        noisy_audio = add_white_noise(audio, snr_db)

        noise = noisy_audio - (
            audio * (
                np.max(np.abs(noisy_audio))
                / np.max(np.abs(audio + (
                    noisy_audio - audio
                )))
            )
        )

        # Calculate SNR (generated noise)
        measured_noise = noisy_audio - audio

        measured_snr = 10 * np.log10(
            np.mean(audio ** 2) /
            np.mean(measured_noise ** 2)
        )
        output_path = f"examples/test_noise_{snr_db}db.wav"
        sf.write(output_path, noisy_audio, sample_rate, subtype="PCM_16")

        print(
            f"Target SNR: {snr_db} dB | "
            f"Measured SNR: {measured_snr:.2f} dB | "
            f"Saved: {output_path}"
        )