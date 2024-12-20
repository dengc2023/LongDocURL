import os
import json
import fitz
from PIL import Image
import sys 
from tqdm import tqdm
import argparse


def get_pdf_filename(pdf_paths_txt, **kwargs):
    with open(pdf_paths_txt, "r", encoding="utf-8") as file:
        files = [line.strip() for line in file.readlines()]
    files = [(filename, filename.split("/")[-2], filename.split("/")[-1][:-4]) for filename in files]
    print("pdf file cnt: ", len(files))
    if "start_id" in kwargs and "end_id" in kwargs:
        start_id = int(kwargs.pop("start_id"))
        end_id = int(kwargs.pop("end_id"))
        return files[start_id:end_id]
    else:
        return files


def save_png(page, pp_no, doc_no, dir, zip_no):
    ddir = f"{dir}/{zip_no}"
    if not os.path.exists(ddir):
        os.makedirs(ddir)
    png_path = ddir + f"/{doc_no}_{pp_no}.png"
    pix = page.get_pixmap()
    pix.save(png_path)

    return png_path


def resize(img_size, rectangle):
    ori_w, ori_h = img_size
    w_rate = ori_w # / 224
    h_rate = ori_h # / 224
    return [round(rectangle[0]/w_rate, 3), round(rectangle[1]/h_rate, 3), round(rectangle[2]/w_rate, 3), round(rectangle[3]/h_rate, 3)]


def extracts(pdf_path, img_save_dir, json_dir):
    doc_no = pdf_path.split("/")[-1][:7] # 4000001.pdf
    zip_no = doc_no[:4]
    filename = pdf_path
    try:
        pages = []
        doc = fitz.open(filename) 
        for pp in doc:
            pages.append(pp)
    except:
        print("error: ", filename.split('/')[-1])
        return  

    lines = []
    for idx, page in enumerate(pages):
        try:
            png_path = save_png(page, idx, doc_no, img_save_dir, zip_no)
            my_image = Image.open(png_path)
        except:
            print("img error: ", filename.split('/')[-1])
            continue 
        
        word_lists = page.get_text("words")
        if not word_lists:
            continue

        for dd in word_lists:
            coordi = resize(my_image.size, list(dd[:4]))
            lines.append({"coordi": coordi, 
                          "word": dd[4].encode('utf-8', 'ignore').decode('utf-8'), 
                          "line_no": dd[5],
                          "block_no": dd[6], 
                          "word_no": dd[7],
                          "page_no": idx})
        
        my_image.close()

    out_data = {
            "zip_no": zip_no, 
            "doc_no": doc_no, 
            "pdf_path": filename,
            "img_size": my_image.size,
            "contents": lines
        }
        
    if not os.path.exists(os.path.join(json_dir, zip_no)):
        os.makedirs(os.path.join(json_dir, zip_no))
    with open(os.path.join(json_dir, zip_no, f"{doc_no}.json"), 'w') as file: 
        json.dump(out_data, file, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, default="/mnt/achao/Downloads/ccpdf_zip/")
    parser.add_argument("--img_save_dir", type=str, default="/mnt/achao/Downloads/pdf_pngs/")
    parser.add_argument("--json_dir", type=str, default="/mnt/achao/Downloads/pdf_jsons/")
    args = parser.parse_args()

    pdf_dir = args.pdf_dir
    img_save_dir = args.img_save_dir
    json_dir = args.json_dir
    pdf_paths = [os.path.join(pdf_dir, pdf_name) for pdf_name in os.listdir(pdf_dir)]
    
    for pdf_path in tqdm(pdf_paths, desc="Processing ", total=len(pdf_paths), unit="item"):
        extracts(pdf_path, img_save_dir, json_dir)
