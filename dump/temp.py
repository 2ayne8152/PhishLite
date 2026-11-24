from load import *

full = load_dataset()
has_path = full["qty_slash_directory"] != -1
print(full.groupby(has_path)["phishing"].agg(["mean", "size"]))