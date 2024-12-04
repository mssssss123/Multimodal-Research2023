import argparse
import json
import parser
import pandas as pd

from src.surrounding_text.ClueWeb22Api import ClueWeb22Api
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data_file_path", type=str, default='/data3/clueweb-anchor/raw_image_text.parquet', help="Path to data file")
    parser.add_argument("--output_file_path", type=str, default='/data3/clueweb-anchor/anchor.json', help="Path to data file")

    args = parser.parse_args()

    input_data = pd.read_parquet(args.data_file_path)

    cw22id_list = input_data['cw22id'].drop_duplicates().values.tolist()
    root_path = '/data2/clueweb'
    all_data_list = []
    for cw22id in tqdm(cw22id_list, desc="Processing"):
        try:
            clueweb_api = ClueWeb22Api(cw22id, root_path)
            inlink = clueweb_api.get_inlinks()
            if inlink:
                try:
                    inlink_dict = json.loads(inlink)
                    all_data_list.append(inlink_dict)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON for cw22id {cw22id}: {e}")
        except Exception as ex:
            print(f"Error creating ClueWeb22Api object for cw22id {cw22id}: {ex}")
    with open(args.output_file_path, "w") as json_file:
        json.dump(all_data_list, json_file)
    print('finish')



if __name__ == '__main__':
    main()
