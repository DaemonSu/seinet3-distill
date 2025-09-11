import os
import numpy as np

def estimate_snr_oracle(iq_data, noise_power_baseline):
    total_power = np.mean(np.abs(iq_data)**2)
    signal_power = max(total_power - noise_power_baseline, 1e-20)
    snr_db = 10 * np.log10(signal_power / noise_power_baseline)
    return snr_db, signal_power, noise_power_baseline

def adjust_snr_oracle(iq_data, noise_power_baseline, target_snr_db):
    total_power = np.mean(np.abs(iq_data)**2)
    current_signal_power = max(total_power - noise_power_baseline, 1e-20)

    target_signal_power = noise_power_baseline * 10**(target_snr_db / 10)

    scale = np.sqrt(target_signal_power / current_signal_power)

    adjusted_iq = iq_data * scale
    return adjusted_iq

def batch_adjust_snr(input_dir, output_dir, noise_power_baseline, target_snr_db):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".sigmf-data"):
            continue

        file_path = os.path.join(input_dir, filename)
        print(f"Processing file: {filename}")

        iq_data = np.fromfile(file_path, dtype=np.complex128)

        snr_before, sig_p_before, noise_p = estimate_snr_oracle(iq_data, noise_power_baseline)
        print(f"  Before adjust: SNR = {snr_before:.2f} dB")

        adjusted_iq = adjust_snr_oracle(iq_data, noise_power_baseline, target_snr_db)

        snr_after, sig_p_after, _ = estimate_snr_oracle(adjusted_iq, noise_power_baseline)
        print(f"  After adjust:  SNR = {snr_after:.2f} dB")

        output_path = os.path.join(output_dir, filename)
        adjusted_iq.tofile(output_path)

if __name__ == "__main__":
    input_dir = r"G:/博士科研/一汽课题/论文阅读/综述/131/neu_m044q5210/KRI-16Devices-RawData/32ft"
    output_dir = r"G:/博士科研/一汽课题/论文阅读/综述/131/neu_m044q5210/KRI-16Devices-RawData/32ft-10-db"
    noise_power_baseline = 1e-7    # 需你根据实际情况调整
    target_snr_db = 1.0              # 目标SNR，比如6dB

    batch_adjust_snr(input_dir, output_dir, noise_power_baseline, target_snr_db)
