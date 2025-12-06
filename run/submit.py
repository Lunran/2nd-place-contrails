import argparse
import gc

import pandas as pd
from tqdm import tqdm

from src_inference1.CoaT_ULSTM import CoaT_ULSTM

# from BaseCoatULSTM import BaseCoatULSTM
from src_inference1.data import ContrailsDataset, rle_encode_less_memory
from src_inference1.fastai_fix import *
from src_inference1.ResNet18_Simple import ResNet18_Simple
from src_inference1.ResNet18_U import ResNet18_U
from src_inference1.ResNet18_ULSTM import ResNet18_ULSTM
from src_inference1.CoaT_ULSTM import CoaT_ULSTM

BS = 1

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="experiments/CoaT_ULSTM.pth", help="Path to the model file")
parser.add_argument("--threshold", type=float, default=0.38, help="Threshold for binary classification")
parser.add_argument("--data_path", type=str, default="data/test", help="Path to the test data")
parser.add_argument(
    "--weights_path", type=str, default="submit_model/coat_lite_medium_384x384_f9129688.pth", help="Path to weights file"
)
parser.add_argument("--output_path", type=str, default="./submission.csv", help="Path to output file")
args = parser.parse_args()

ds = ContrailsDataset(args.data_path)
dl = DataLoader(ds, BS, shuffle=False, num_workers=min(2, BS))

MODELS = []
MODELS += [(args.model_path, CoaT_ULSTM, 1)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models, weights = [], []
for path, Model, w in MODELS:
    model = Model(weights_path=args.weights_path)
    if isinstance(model, list):
        model = model[0]
    if path is not None:
        state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        model.load_state_dict(state_dict)
        del state_dict
    model.to(device).eval()
    models.append(model)
    weights.append(w)
gc.collect()

names, preds = [], []
print("-> Inference ...\n")
for x, y in tqdm(dl):
    with torch.no_grad():
        x = x.to(device)
        p = torch.stack([m(x).sigmoid() * w for m, w in zip(models, weights)], 0).sum(0).squeeze(1) / sum(weights)
    p = p.cpu().numpy()

    for pi, yi in zip(p, y):
        rle = rle_encode_less_memory(pi > args.threshold)
        if len(rle) == 0:
            rle = "-"
        names.append(yi)
        preds.append(rle)

print("-> Inference completed ! \n")

df = pd.DataFrame({"record_id": names, "encoded_pixels": preds})
df.to_csv(args.output_path, index=False)
print(df.head())
