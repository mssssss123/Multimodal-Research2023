import argparse
import base64
import io
import os
import warnings

import clip
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='/data4/clip-pretrain/clue_webdata',
                        help="Path to data file")
    parser.add_argument("--output_path", type=str, default='/data1/lvyuanhuiyi/meisen/test_demo',
                        help="Path to data file")
    parser.add_argument("--file_num", type=int, default=1, help="Path to data file")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    image_path = f"image_{args.file_num}"
    image_path = os.path.join(args.output_path, image_path)
    os.makedirs(image_path, exist_ok=True)

    parquet_files = [file for file in os.listdir(args.input_path) if file.endswith('.parquet')]
    parquet_files.sort()
    selected_file = parquet_files[args.file_num]

    input_file_path = os.path.join(args.input_path, selected_file)
    df = pd.read_parquet(input_file_path)

    for index, row in df.iterrows():
        try:
            image_type = row['IMG_TYPE']
            cur_image = Image.open(io.BytesIO(row['BUFFER'])).convert("RGB")
            save_img_path = os.path.join(image_path, f'{index}.{image_type.lower()}')
            cur_image.save(save_img_path)
            df.at[index, 'IMAGE_PATH'] = save_img_path
        except Exception as e:
            print(f"Error processing image {index}: {e}")



    selected_columns = ['alt', 'surround', 'IMAGE_PATH']
    new_df = df[selected_columns]
    parquet_path = f"text_{args.file_num}.parquet"
    parquet_path = os.path.join(args.output_path, parquet_path)
    new_df.to_parquet(parquet_path)




    print('finish')

if __name__ == '__main__':
    main()