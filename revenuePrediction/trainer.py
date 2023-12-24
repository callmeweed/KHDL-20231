import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import RevenuePredictor
from transformers import RobertaModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import features_config
from dataset import CustomDataset
import json
import pandas as pd


def preprocess():
    df = pd.read_csv("C:\Users\hoanganh\Desktop\DS\KHDL-20231\pre_process\proccessed\movie.csv")

    # df = df[(df["revenue"] != 0) & (df["budget"] != 0)]
    # df["percent"] = df["revenue"]/df["budget"]
    # df = df[(df["percent"] > 0.1) & (df["percent"] < 10)]
    df = df[(df["revenue"] > 10000) & (df["revenue"] < 4e6)]

    df["Director"] = df["Director"].apply(lambda x: json.loads(x)[0] if len(json.loads(x)) > 0 else 0)
    df["Writers"] = df["Writers"].apply(lambda x: json.loads(x)[0] if len(json.loads(x)) > 0 else 0)
    df["country_of_origin"] = df["country_of_origin"].apply(lambda x: json.loads(x)[0] if len(json.loads(x)) > 0 else 0)
    df["languages"] = df["languages"].apply(lambda x: json.loads(x)[0] if len(json.loads(x)) > 0 else 0)

    df["originalTitle"].fillna("<unk>", inplace=True)
    df["summary"].fillna("<unk>", inplace=True)
    df["concatenated"] = df.apply(lambda x: x.originalTitle + "; " + x.summary, axis=1)

    scaler = StandardScaler()
    df["runtimeMinutes"] = scaler.fit_transform(df.runtimeMinutes.to_numpy().reshape(-1,1)).reshape(len(df),)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df["revenue"] = scaler.fit_transform(df.revenue.to_numpy().reshape(-1,1)).reshape(len(df),)
    df["budget"] = scaler.fit_transform(df.budget.to_numpy().reshape(-1,1)).reshape(len(df),)

    genres = []
    genres_mask = []
    company = []
    company_mask = []
    cast = []
    cast_mask = []

    def pad_trunc_mask(value, feature):
        f_cf = features_config["categorical"]["multi"][feature]
        value = json.loads(value)[:f_cf["max_seq_len"]]
        att_mask = [1]*len(value)
        if len(value) < f_cf["max_seq_len"]:
            att_mask += (f_cf["max_seq_len"] - len(value))*[0]
            value += (f_cf["max_seq_len"] - len(value))*[f_cf["vocab_size"]-1]
        att_mask[0]=1
        return value, att_mask, max(value), min(value)

    max_g, max_c, max_ca = 0, 0, 0
    min_g, min_c, min_ca = 1e7, 1e7, 1e7

    for i in range(len(df)):
        row = df.iloc[i]
        g_v, g_mask, g, _g = pad_trunc_mask(row.genres, "genre")
        if g > max_g: max_g = g
        if _g < min_g: min_g = _g
        genres.append(g_v)
        genres_mask.append(g_mask)
        c_v, c_mask, c, _c = pad_trunc_mask(row.production_companies, "company")
        if c > max_c: max_c = c
        if _c < min_c: min_c = _c
        company.append(c_v)
        company_mask.append(c_mask)
        ca_v, ca_mask, ca, _ca = pad_trunc_mask(row.Stars, "cast")
        if ca > max_ca: max_ca = ca
        if _ca < min_ca: min_ca = _ca
        cast.append(ca_v)
        cast_mask.append(ca_mask)

    df["genres"] = genres
    df["genres_mask"] = genres_mask
    df["production_companies"] = company
    df["company_mask"] = company_mask
    df["cast"] = cast
    df["cast_mask"] = cast_mask
    print(max_g, max_c, max_ca)
    print(min_g, min_c, min_ca)

    return df

if __name__ == "__main__":
    device="cuda"

    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly(True)
    model = RevenuePredictor(RobertaModel.from_pretrained('roberta-base'), features_config)
    # model.load_state_dict(torch.load("source/weights/cp_vision_encoder_decoder_augmented_data.pt"))
    model.to(device)
    # model.collection_encoder.to(device)

    #data loader
    # anot = pd.read_csv("concated_anot.csv").sample(frac=1, random_state=0).reset_index(drop=True)
    df = preprocess()
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    traindataset = CustomDataset(df[:-5000], features_config)
    trainloader = DataLoader(dataset=traindataset, batch_size=64, shuffle=True, generator=torch.Generator(device="cpu"))
    valdataset = CustomDataset(df[-5000:], features_config)
    valloader = DataLoader(dataset=valdataset, batch_size=64, shuffle=False, generator=torch.Generator(device="cpu"))

    #optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.0001}], weight_decay=1e-5)


    #number of epochs
    n_epoch = 15

    # wandb.login(key = 'e0e0a2547f255a36f551d7b6a166b84e5139d276')
    # wandb.init(
    #   project="revenue_prediction",
    #   name=f"abc",
    #   config={
    #       "learning_rate": 0.0001,
    #       "architecture": "v_t",
    #       "dataset": "gen",
    #       "epochs": 10,
    #   }
    # )


    print("Start .....")
    best_loss = 1e6
    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    for epoch in range(n_epoch):
        print(f"epoch {epoch} ----------------------------")
        model.train()
        train_loss = 0
        
        for batch in tqdm(trainloader):

            optimizer.zero_grad()
            
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            
    #         print(model(batch))
            # forward
            
            
    #         if loss.isnan(): loss = torch.rand(1, require_grad=True)/10000
    #         try:
            out, att_weights = model(batch)
            loss = loss_fn(out, batch["rev"].reshape(batch["rev"].shape[0],))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 4)

            optimizer.step()

            train_loss += loss.item()
    #         except:
    #             print(batch["idx"])

        else:
            print(f"train loss: {train_loss/len(trainloader)}")
            # torch.save(model.state_dict(), "source/weights/last_vgg_transformer.pt")
            with torch.no_grad():
                model.eval()
                val_loss = 0
                for batch in tqdm(valloader):
                    for key in batch.keys():
                        batch[key]=batch[key].to(device)
                    out, att_weights = model(batch)
                    loss = loss_fn(out, batch["rev"].reshape(batch["rev"].shape[0],))
                    val_loss += loss
                if best_loss > val_loss/len(valloader):
                    best_loss = val_loss/len(valloader)
                    torch.save(model.state_dict(), "checkpoint/cp.pt")
    #             wandb.log({"train_loss": train_loss/len(trainloader), "val_loss": val_loss/len(valloader)})
                print(f"val loss: {val_loss/len(valloader)}")
    # wandb.finish()