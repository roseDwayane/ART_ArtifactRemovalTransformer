"""Offline EEG artifact-removal pipeline.

Reads a multichannel EEG csv, runs one of the four trained models, and
writes the denoised signal back as csv. Configure the parameters below,
then run from the repository root:

    python main.py
"""

import os

import art


def main():
    # ---- configure your run here -------------------------------------------
    input_path = os.path.join('.', 'sampledata')
    input_name = 'sampledata.csv'
    sample_rate = 256                       # Hz of the input csv
    modelname = 'ART'                       # 'ART' | 'ICUNet' | 'ICUNet++' | 'ICUNet_attn'
    output_path = os.path.join('.', 'sampledata')
    output_name = 'outputsample.csv'
    mapping_name = os.path.join(
        '.', 'sampledata', 'sample_chanlocs_mapping_result.json',
    )
    # ------------------------------------------------------------------------

    input_file = os.path.join(input_path, input_name)
    output_file = os.path.join(output_path, output_name)

    mapping_result, num_channel, num_group = art.read_mapping_result(mapping_name)

    print(f"[INFO] device = {art.DEVICE}")
    print(f"[INFO] model  = {modelname}")
    print(f"[INFO] input  = {input_file}")
    print(f"[INFO] output = {output_file}")
    print(f"[INFO] groups = {num_group} / channels = {num_channel}")

    for i in range(num_group):
        preprocessed = art.preprocessing(input_file, sample_rate, mapping_result[i])
        reconstructed = art.reconstruct(modelname, preprocessed, output_name, i)
        art.postprocessing(
            reconstructed, sample_rate, output_file, mapping_result[i], i, num_channel,
        )

    print(f"[DONE] Reconstructed signal saved to {output_file}")


if __name__ == '__main__':
    main()
