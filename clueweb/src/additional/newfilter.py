import argparse
import base64
import io
import warnings

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from PIL import Image
import open_clip


def is_valid_image(image_bytes,preprocess):
    try:
        with warnings.catch_warnings(record=True) as w:
            image_buffer = io.BytesIO(image_bytes)
            images = preprocess(Image.open(image_buffer))
            for warning in w:
                if issubclass(warning.category, UserWarning):
                    return False  # 过滤掉触发警告的图像

        return True  # 图像有效（没有异常或警告）
    except Exception:
        return False  # 过滤掉触发异常的图像


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_file_path", type=str, default='/data3/clueweb-anchor/raw_image_text.parquet',
                        help="Path to data file")
    parser.add_argument("--output_file_path", type=str, default='/data1/meisen/multi-modal/open_clip-main/dataset/filter_image_text.parquet',
                        help="Path to data file")
    parser.add_argument("--clip_path", type=str, default='/data1/meisen/multi-modal/open_clip-main/src/checkpoints/Model-B-32_Data-400M_Samples-13B_lr-1e-3_bs-86k.pt',
                        help="Path to data file")

    args = parser.parse_args()
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=args.clip_path)

    table = pq.read_table(args.parquet_file_path)
    df = table.to_pandas()

    # 使用 is_valid_image 函数过滤受损图像
    df['is_valid'] = df['BUFFER'].apply(lambda image_buffer: is_valid_image(image_buffer, preprocess))
    filtered_df = df[df['is_valid']]

    # 将过滤后的数据保存到新的Parquet文件
    filtered_table = pa.Table.from_pandas(filtered_df)
    pq.write_table(filtered_table, args.output_file_path)

    num_filtered = len(df) - len(filtered_df)
    print(f"过滤掉的样本数目: {num_filtered}")
    print(f"过滤后的数据已保存到 {args.output_file_path}。")

    print('finish')


if __name__ == '__main__':
    main()
