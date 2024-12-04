import argparse
import io
import json

import pandas as pd
from PIL import Image

def get_one_doc_example(doc_path,index):
    """
    目前这个文件处理的完成度很高，唯一需要做的是对caption-img的相似度做过滤，
    目前处理的代码我已经写好(clip_score.py)，但好像有小bug，可以一起解决一下。
    唯一的担忧在于text document的质量并不能保证,但没有什么好的处理办法。

    """

    input_data = pd.read_parquet(doc_path)
    example = input_data.iloc[index]
    example = example.to_dict()
    # 这个id用于将document从query文件对应找出它的query
    cw22id = example['cw22id']
    # image document
    caption = example['alt']
    image_buffer = io.BytesIO(example['BUFFER'])
    img = Image.open(image_buffer)
    # text document
    text_doc = example['surround']
    return example
def get_one_query_example(qry_file_path,index):
    """
    需要对anchor-text进行过滤，请参考：https://github.com/Veronicium/AnchorDR
    需要将其转化为更方便的形式(可选，不做也没啥事)：（query，pos_text_id,pos_image_id)

    """
    with open(qry_file_path, 'r') as json_file:
        data = json.load(json_file)
        example = data[index]
        # 获取cw22id,可以通过这个去doc文件找到多个query所对应的doc，因此这个数据集是一个多对多的映射关系
        cw22id = example['ClueWeb22-ID']
        # 一个cw22id 对应的有多个query
        anchor_list = example['anchors']
        # 一个例子 关于这里的一些属性解释：
        # Each anchor is a list, containing:
        #   (1) the url of the source webpage,
        #   (2) the url hash of the source webpage,
        #   (3) the anchor text
        #   (4) an indicator of header/footer. "" or "0" means it is not a header/footer
        #   (5) the language of the source webpage
        anchor = anchor_list[index]
        # the anchor text-》query
        query = anchor[2]
        return example



def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--doc_file_path", type=str, default='/data3/clueweb-anchor/raw_image_text.parquet', help="Path to data file")
    parser.add_argument("--qry_file_path", type=str, default='/data3/clueweb-anchor/anchor.json', help="Path to data file")
    args = parser.parse_args()

    # docexample = get_one_doc_example(args.doc_file_path,index=0)
    qryexample = get_one_query_example(args.qry_file_path,index=0)
    print('---')


if __name__ == '__main__':
    main()