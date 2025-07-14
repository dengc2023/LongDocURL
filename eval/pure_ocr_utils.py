import json


def record2text_with_layout(record):
    text = ""
    img_width, img_height = record['docInfo']['pages'][0]['imageWidth'], record['docInfo']['pages'][0]['imageHeight']
    for item in record['layouts']:
        _type, sub_type = item['type'], item['subType']
        item_text = item['text']
        x1y1 = item['pos'][0]; x2y2 = item['pos'][2]
        text += f"(type: {_type}, sub_type: {sub_type}, box: {tuple(float(f'{_:.2f}') for _ in (x1y1['x']/img_width, x1y1['y']/img_height, x2y2['x']/img_width, x2y2['y']/img_height))})" + f" {item_text}\n"

    return text

def record2text(record):
    text = ""
    for item in record['layouts']:
        item_text = item['text']
        text += f"{item_text}\n"

    return text


def get_pure_ocr_prompt_docmind(doc_no: str, **kwargs):
    zip_no = doc_no[:4]
    json_path = "/mnt/achao/Downloads/pdf_jsons/{}/{}_docmind_results.json"
    record = json.load(open(json_path.format(zip_no, doc_no), "r", encoding="utf-8"))['contents']
    ocr_text_template = "page_no: {}\n{}\n\n"
    
    start_page = kwargs.pop("start_page", 0); end_page = kwargs.pop("end_page", start_page+1)
    if "extra_infos" in kwargs and "with_layout" in kwargs["extra_infos"] and kwargs["extra_infos"]["with_layout"]:
        ocr_texts = [record2text_with_layout(record[f"page_{idx}"]) for idx in range(start_page, end_page+1) if f"page_{idx}" in record]
    else:
        ocr_texts = [record2text(record[f"page_{idx}"]) for idx in range(start_page, end_page+1) if f"page_{idx}" in record]
    pages_used = end_page - start_page + 1
    print("number of pages used: ", end_page - start_page + 1)

    ocr_prompt = "\n\n"
    for page_no, ocr_text in zip(range(start_page, end_page+1), ocr_texts):
        ocr_prompt += ocr_text_template.format(page_no+1, ocr_text) # why page_no+1？
        
    return ocr_prompt

def get_pure_ocr_prompt_pymupdf(doc_no: str, **kwargs):
    zip_no = doc_no[:4]
    json_path = "/mnt/achao/Downloads/pdf_jsons/{}/{}_line_level.json"
    ocr_texts_doc = json.load(open(json_path.format(zip_no, doc_no), "r", encoding="utf-8"))["pages_str"] # list
    ocr_text_template = "page_no: {}\n{}\n\n"
    
    start_page = kwargs.pop("start_page", 0); end_page = kwargs.pop("end_page", start_page+1)

    pages_used = end_page - start_page + 1
    print("number of pages used: ", end_page - start_page + 1)

    ocr_prompt = "\n\n"
    for page_no, ocr_texts_page in zip(range(start_page, end_page+1), ocr_texts_doc):
        ocr_prompt += ocr_text_template.format(page_no+1, ocr_texts_page) # why page_no+1?
        
    return ocr_prompt