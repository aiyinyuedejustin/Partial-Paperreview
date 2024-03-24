import scipdf
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type=str, default=None, help="The path of the folder about paper pdf.")
args = parser.parse_args()

org_dir_path = Path(args.dir_path).resolve()
trg_dir_path = org_dir_path.with_name("scipdf_parser_results")

error_log = {}

if not trg_dir_path.exists():
    trg_dir_path.mkdir()

for pdf_file in tqdm(org_dir_path.glob("*.pdf")):
    trg_path = trg_dir_path.joinpath(pdf_file.name).with_suffix(".json")
    if trg_path.exists():
        continue
    try:
        article_dict = scipdf.parse_pdf_to_dict(str(pdf_file)) # return dictionary
        with open(trg_path, "w") as f:
            json.dump(article_dict, f)
    except Exception as e:
        error_log[str(pdf_file.name)] = str(e)
        continue

error_log_path = trg_dir_path.with_name("error_log_scipdf.json")
with open(error_log_path, "w") as fe:
    json.dump(error_log, fe)